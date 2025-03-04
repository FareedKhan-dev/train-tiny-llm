import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List

from .LMConfig import LMConfig  # Assuming LMConfig is defined in a local module


class RMSNorm(torch.nn.Module):
    """
    Implements Root Mean Square Normalization (RMSNorm).
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter

    def forward(self, x):
        # Normalize the input tensor using RMSNorm formula.
        # Uses .float() for FP32 precision during calculation, then casts back.
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    Precomputes complex exponentials (cis) for rotary positional embeddings.

    Args:
        dim: Dimensionality of the embeddings.
        end: Maximum sequence length.
        theta: Scaling factor for frequencies.

    Returns:
        torch.Tensor: Precomputed complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore # Sequence indices
    freqs = torch.outer(t, freqs).float()  # type: ignore # Outer product to get frequencies for each position
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 # Create complex exponentials
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    """
    Applies rotary positional embeddings to query (xq) and key (xk) tensors.

    Args:
        xq: Query tensor.
        xk: Key tensor.
        pos_cis: Precomputed complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Query and key tensors with rotary embeddings applied.
    """

    def unite_shape(pos_cis, x):
        # Reshape pos_cis to have compatible dimensions with x for broadcasting.
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # Reshape and convert to complex numbers for efficient multiplication.
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)  # Reshape pos_cis for broadcasting
    # Apply rotary embeddings via complex number multiplication.
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)  # Ensure output type matches input


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key-value pairs for multi-query attention.

    Args:
        x: Key-value tensor.
        n_rep: Number of times to repeat each head.

    Returns:
        torch.Tensor: Key-value tensor with repeated heads.
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # Expand and reshape to repeat the key-value heads.
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Implements multi-head attention with rotary positional embeddings.
    """

    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads  # Total number of attention heads
        self.n_local_kv_heads = self.n_kv_heads  # Number of key-value attention heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # Repetition factor for key-value heads
        self.head_dim = args.dim // args.n_heads  # Dimension of each attention head
        # Linear projections for query, key, and value.
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # Output projection
        self.attn_dropout = nn.Dropout(args.dropout)  # Dropout for attention weights
        self.resid_dropout = nn.Dropout(args.dropout)  # Dropout for the residual connection
        self.dropout = args.dropout
        # Check for Flash Attention availability (requires PyTorch >= 2.0).
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # Causal mask for autoregressive decoding.
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)  # Upper triangular mask
        self.register_buffer("mask", mask, persistent=False)  # Register as a buffer (not a parameter)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        # Apply linear projections to get query, key, and value.
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Reshape for multi-head attention.
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # Apply rotary positional embeddings.
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # KV Cache implementation
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # Concatenate with cached keys
            xv = torch.cat([past_key_value[1], xv], dim=1)  # Concatenate with cached values
        past_kv = (xk, xv) if use_cache else None  # Store current keys and values for caching

        # Repeat key-value pairs for multi-query attention.
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # Attention mechanism selection: Flash Attention (if available) or standard attention.
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0  # Dropout only during training
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,  # No explicit mask needed (causal masking is handled internally)
                dropout_p=dropout_p,
                is_causal=True  # Enforce causal attention
            )
        else:
            # Standard attention
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # Calculate attention scores
            scores += self.mask[:, :, :seq_len, :seq_len]  # Apply causal mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # Softmax to get attention weights
            scores = self.attn_dropout(scores)  # Apply dropout
            output = scores @ xv  # Weighted sum of values

        # Reshape and apply output projection.
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    Implements the feedforward network (FFN) used in each transformer block.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        # Compute hidden dimension for FFN.
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        # Linear layers with SiLU activation.
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)  # Dropout after the FFN

    def forward(self, x):
        # Apply FFN transformation:  x -> SiLU(xW1) * (xW3) -> (result)W2 -> dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))



class TransformerBlock(nn.Module):
    """
    Implements a single transformer block.
    """

    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)  # Multi-head attention

        self.layer_id = layer_id
        # Layer normalization for attention and FFN.
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # Feedforward network.
        self.feed_forward = FeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        # Attention block with residual connection.
        h_attn, past_kv = self.attention(
            self.attention_norm(x),  # Normalize input before attention
            pos_cis,  # Rotary positional embeddings
            past_key_value=past_key_value,  # Pass cached key-value pairs
            use_cache=use_cache  # Whether to use caching
        )
        h = x + h_attn  # Residual connection
        # Feedforward block with residual connection.
        out = h + self.feed_forward(self.ffn_norm(h))  # Normalize input before FFN
        return out, past_kv


class TransformerLM(PreTrainedModel):
    """
    The main Transformer language model.
    """
    config_class = LMConfig  # Use LMConfig as the configuration class

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()  # Use default config if none provided
        super().__init__(self.params)  # Initialize PreTrainedModel
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        # Token embeddings.
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)  # Dropout after embeddings
        # Transformer blocks.
        self.layers = nn.ModuleList([TransformerBlock(l, params) for l in range(self.n_layers)])
        # Final layer normalization.
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # Output layer (maps from hidden states to logits).
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # Tie token embeddings and output weights.
        self.tok_embeddings.weight = self.output.weight
        # Precompute and register rotary positional embeddings.
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()  # Use CausalLMOutputWithPast for output

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):

        past_key_values = past_key_values or [None] * len(self.layers)  # Initialize empty cache if None
        start_pos = args.get('start_pos', 0)  # Get starting position for sequence generation

        h = self.dropout(self.tok_embeddings(input_ids))  # Get token embeddings and apply dropout
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]  # Get relevant rotary embeddings

        past_kvs = []  # Store past key-value pairs for caching
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],  # Pass cached key-value pairs
                use_cache=use_cache
            )
            past_kvs.append(past_kv)  # Store updated key-value pairs

        logits = self.output(self.norm(h))  # Final layer normalization and output projection

        # Set output attributes using __setitem__
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT  # Return CausalLMOutputWithPast object

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        """
        Generates text from the model.

        Args:
            input_ids: Initial input token IDs.
            eos_token_id: End-of-sequence token ID.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling probability.
            stream: Whether to stream the output (yield tokens one by one).
            rp: Repetition penalty.
            use_cache: Whether to use key-value caching.
            pad_token_id: pad token id.
            **args: Additional arguments passed to the forward method.

        Returns:
            torch.Tensor: Generated sequence of token IDs.
                         (if stream=False)
            Generator[torch.Tensor, None, None]:  Generated tokens.
                                        (if stream=True)
        """
        # Stream generation
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # Direct generation (collects all tokens at once)
        generated = []
        for i in range(input_ids.size(0)):
            # remove padding
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            # generate tokens
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # collect the generated token one-by-one
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            # concat the input and generated tokens together
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        # find the longest sequence
        max_length = max(seq.size(1) for seq in generated)
        # padding the sequences
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        # concatenate all generated tensors together
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        """
        Helper function for streaming text generation.
        """
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                # For the first sequence or when not using cache, process the entire input sequence.
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                # For subsequent sequences with caching, process only the last token.
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values  # Get logits and updated cache

            # Apply repetition penalty.
            logits[:, list(set(input_ids.tolist()[0]))] /= rp

            # Apply temperature scaling.
            logits /= (temperature + 1e-9)

            # Apply top-p (nucleus) sampling.
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')  # Set probabilities to -inf for filtered tokens

            # Sample the next token.
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)  # Append the new token to the sequence
            yield input_ids[:, start:]  # Yield the generated tokens (excluding the initial input)

            # Break if EOS token is generated.
            if input_ids_next.item() == eos_token_id:
                break