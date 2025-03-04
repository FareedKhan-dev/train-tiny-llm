import os
import platform
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from transformer.model import TransformerLM
from transformer.LMConfig import LMConfig
from transformer.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """Prints the content if not in DDP mode or if it's the main process."""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """Calculates the learning rate using a cosine annealing schedule.

    Args:
        current_step: The current training step.
        total_steps: The total number of training steps.
        lr: The initial learning rate.

    Returns:
        The calculated learning rate.
    """
    # Linear warmup for the first 10% of steps, followed by cosine decay.
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """Trains the model for one epoch.

    Args:
        epoch: The current epoch number.
        wandb: The wandb object for logging (or None if not using wandb).
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # Loss function for causal language modeling
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # Move data to the specified device.
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Cosine Annealing Learning Rate Scheduler
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass with automatic mixed precision (if enabled).
        with ctx:
            res = model(X)
            # Calculate the loss.  The loss is masked to ignore padding tokens.
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            #loss += res.aux_loss # Removed as there's no aux_loss now
            loss = loss / args.accumulation_steps  # Scale loss for gradient accumulation

        # Backward pass and gradient scaling.
        scaler.scale(loss).backward()

        # Gradient accumulation and optimization step.
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # Clip gradients

            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the gradient scaler

            optimizer.zero_grad(set_to_none=True)  # Reset gradients

        # Logging.
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # Scale back the loss for reporting
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))  # Calculate epoch time

            # Log to wandb (if enabled and in the main process).
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # Model saving.
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # Set the model to evaluation mode
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}.pth' #Simplified naming

            # Save the model state dictionary.  Handle DDP models correctly.
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()  # Set the model back to training mode


def init_model(lm_config):
    """Initializes the model and tokenizer.

    Args:
        lm_config: The language model configuration.

    Returns:
        The initialized model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = TransformerLM(lm_config).to(args.device)
    Logger(f'LLM total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} Million')
    return model, tokenizer


def init_distributed_mode():
    """Initializes distributed training mode."""
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # Not used, consider removing
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# Example usage: torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer LLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # If you want the fastest zero implementation, set epochs to 1; otherwise, you should train 2-6 epochs with limited data.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")  # Not used, consider removing or implementing its use
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Transformer-Pretrain")  # Updated project name
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)  # Not explicitly used, consider removing
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)  # Used by torchrun
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="custom_tokenizer")
    args = parser.parse_args()

    # Create LMConfig.
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)  # Redundant, but harmless
    tokens_per_iter = args.batch_size * lm_config.max_seq_len # calculated but never used
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"Transformer-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # Use nullcontext for CPU, torch.cuda.amp.autocast() for mixed precision.
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # Check if running in DDP mode.
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # Initialize wandb (if enabled and in the main process).
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # Initialize model and tokenizer.
    model, tokenizer = init_model(lm_config)

    # Create dataset and dataloader.  Use DistributedSampler for DDP.
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,  # Don't drop the last batch (important for consistent evaluation)
        shuffle=(train_sampler is None),  # Shuffle only if not using a DistributedSampler
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # Initialize gradient scaler and optimizer.
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16'])) # dtype check, but dtype isn't used.  Consider removing
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Handle DDP.  Exclude pos_cis from DDP synchronization.
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        if train_sampler is not None:  # Important for DDP: set the epoch for the sampler
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb)