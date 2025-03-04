import os
import argparse
import time
import math
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from transformer.model import TransformerLM
from transformer.LMConfig import LMConfig
from transformer.dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """Prints content only for the main process in DDP."""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """Calculates learning rate with cosine annealing."""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """Trains the model for one epoch."""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # Move tensors to device
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Cosine Annealing Learning Rate Scheduler
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass (with mixed precision if enabled)
        with ctx:
            res = model(X)
            # Calculate loss, apply mask, and average
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # loss += res.aux_loss # Removed aux_loss
            loss = loss / args.accumulation_steps  # For gradient accumulation

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps, # show scaled loss value
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # Wandb logging (if enabled and main process)
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # Model saving (if enabled and main process)
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            # ckp = f'{args.save_dir}/full_sft_{lm_config.dim}.pth'  # Simplified checkpoint name
            ckp = os.path.join(args.save_dir, f'sft_{lm_config.dim}.pth')

            # Save model state dict (handle DDP)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """Initializes the model, loads pretrained weights, and moves to device."""
    tokenizer = AutoTokenizer.from_pretrained('./model/transformer_tokenizer')  # Use consistent naming
    model = TransformerLM(lm_config)  # Use TransformerLM
    # Load pretrained weights
    ckp = os.path.join(args.out_dir, f'pretrain_{lm_config.dim}.pth') # Load from pretrain checkpoint.
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False) # strict=False is crucial if loading a pretrained model to sft
    Logger(f'LLM total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} Million')
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    """Initializes distributed training."""
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer LLM SFT")
    parser.add_argument("--out_dir", type=str, default="out") # for loading pretrained model
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")  # Consider removing or using
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Transformer-SFT")  # Updated project name
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)  # Not explicitly used
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)  # For torchrun
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    # parser.add_argument('--use_moe', default=False, type=bool) # Removed MoE
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")  # Example path

    args = parser.parse_args()

    # Create LMConfig (no MoE)
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len)
    args.save_dir = os.path.join(args.out_dir, "sft_checkpoints") # save sft model in different directory
    os.makedirs(args.save_dir, exist_ok=True)
    # os.makedirs(args.out_dir, exist_ok=True) # No need as out_dir is exist
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"Transformer-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # Context for mixed precision
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # Check for DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # Wandb initialization (if enabled and main process)
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # Initialize model and tokenizer
    model, tokenizer = init_model(lm_config)

    # Create dataset and dataloader (with DDP sampler if needed)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,  # Important for consistent eval, especially with DDP
        shuffle=(train_sampler is None),  # Shuffle only if not using DistributedSampler
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # Gradient scaler and optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16'])) # dtype check
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # DDP setup (exclude pos_cis)
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        if train_sampler is not None:  # CRUCIAL for DDP: set epoch for sampler
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb)