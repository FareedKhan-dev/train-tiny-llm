from transformers import PretrainedConfig

class LMConfig(PretrainedConfig):
    """
    Configuration class for the language model.  Inherits from PretrainedConfig.
    This configuration class stores all the hyperparameters of the model.

    Attributes:
        model_type (str): Model type identifier.
        dim (int): Embedding dimension.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of key-value attention heads (for multi-query attention).
        vocab_size (int): Size of the vocabulary.
        hidden_dim (int, optional): Hidden dimension of the feedforward network.
            If None, it's calculated based on `dim` and `multiple_of`.
        multiple_of (int): Used to calculate `hidden_dim` if `hidden_dim` is None.
        norm_eps (float): Epsilon value for layer normalization.
        max_seq_len (int): Maximum sequence length.
        rope_theta (int): Theta value for rotary positional embeddings.
        dropout (float): Dropout probability.
        flash_attn (bool): Whether to use Flash Attention (if available).
        use_moe (bool): Whether to use Mixture of Experts.
        num_experts_per_tok (int): Number of experts per token (only if use_moe is True).
        n_routed_experts (int): Total number of experts (only if use_moe is True).
        n_shared_experts (bool): Whether to use shared experts (only if use_moe is True).
        scoring_func (str): Scoring function for expert selection (only if use_moe is True).
        aux_loss_alpha (float): Weight for the auxiliary loss (only if use_moe is True).
        seq_aux (bool): Whether to compute aux loss at sequence level (only if use_moe is True).
        norm_topk_prob (bool): Whether to normalize top-k probabilities (only if use_moe is True).
    """

    model_type = "transformerlm"

    def __init__(
        self,
        dim: int = 512,             # Embedding dimension
        n_layers: int = 8,           # Number of transformer layers
        n_heads: int = 8,           # Number of attention heads
        n_kv_heads: int = 2,        # Number of key-value heads (multi-query attention)
        vocab_size: int = 6400,     # Size of the vocabulary
        hidden_dim: int = None,     # Hidden dimension of the FFN (calculated if None)
        multiple_of: int = 64,      # Used for calculating hidden_dim
        norm_eps: float = 1e-5,     # Epsilon value for layer normalization
        max_seq_len: int = 8192,    # Maximum sequence length
        rope_theta: int = 1e6,      # Theta for RoPE
        dropout: float = 0.0,       # Dropout probability
        flash_attn: bool = True,    # Use Flash Attention if available
        num_experts_per_tok: int = 2, # Number of experts per token (unused)
        n_routed_experts: int = 4,   # Total number of experts (unused)
        n_shared_experts: bool = True,# Use shared experts (unused)
        scoring_func: str = "softmax",# Expert scoring function (unused)
        aux_loss_alpha: float = 0.1, # Weight for auxiliary loss (unused)
        seq_aux: bool = True,        # Sequence-level auxiliary loss (unused)
        norm_topk_prob: bool = True, # Normalize top-k probabilities (unused)
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        super().__init__(**kwargs)