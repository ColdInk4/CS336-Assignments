import argparse
from cs336_basics.data_loader import get_batch
import numpy as np
from cs336_basics.transformer import Transformer_LM
from cs336_basics.losses import cross_entropy
from cs336_basics.training import load_checkpoint, save_checkpoint
from cs336_basics.optim import AdamW, gradient_clipping, lr_cosine_schedule
from einops import rearrange
import torch
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Transformer Training")

parser.add_argument("--train_path", help="path to train tokenids")
parser.add_argument("--valid_path", help="path to valid tokenids")

parser.add_argument("--d_model", type=int, help="d_model")
parser.add_argument("--vocab_size", type=int, help="vocab_size")
parser.add_argument("--context_length", type=int, help="context_length")
parser.add_argument("--num_layers", type=int, help="num_layers")
parser.add_argument("--num_heads", type=int, help="num_heads")
parser.add_argument("--theta", type=float, help="the ROPE theta")

parser.add_argument("--batch_size", type=int, help="batch_size")
parser.add_argument("--total_train_steps", type=int, help="total train steps")

# parser.add_argument("--dtype", help="dtype")

# parser.add_argument("-lr", "--learning_rate", help="learning rate in adamw")
parser.add_argument(
    "--max_lr", type=float, help="max_learning_rate in lr_cosine_schedule"
)
parser.add_argument(
    "--min_lr", type=float, help="min_learning_rate in lr_cosine_schedule"
)
parser.add_argument(
    "--warmup_steps", type=int, help="warmup_steps in lr_cosine_schedule"
)
parser.add_argument(
    "--cosine_cycle_steps", type=int, help="cosine_cycle_steps in lr_cosine_schedule"
)
parser.add_argument("--betas", type=float, nargs=2, help="beta, need two nums in adamw")
parser.add_argument("--weight_decay", type=float, help="weight decay in adamw")
parser.add_argument("--adamw_eps", type=float, help="eps in adamw")

parser.add_argument(
    "--max_grad_norm", type=float, help="max_l2_norm in gradient_clipping"
)

parser.add_argument(
    "--log_interval", type=int, help="log interval in steps", default=10
)
parser.add_argument(
    "--eval_interval", type=int, help="eval interval in steps", default=10
)

parser.add_argument(
    "--checkpoint_interval", type=int, help="checkpoint interval in steps", default=10
)
parser.add_argument("--resume_config_path", help="config path in")
parser.add_argument("--resume_checkpoint_path", help="checkpoint path in")
parser.add_argument("--checkpoint_dir", help="checkpoint path out", required=True)
parser.add_argument("--device", help="device")

# 解析
args = parser.parse_args()

device = args.device

log_interval = args.log_interval
eval_interval = args.eval_interval

checkpoint_interval = args.checkpoint_interval
resume_config_path = args.resume_config_path
resume_checkpoint_path = args.resume_checkpoint_path
checkpoint_dir = args.checkpoint_dir

if bool(resume_checkpoint_path) != bool(resume_config_path):
    raise ValueError(
        "resume_checkpoint_path and resume_config_path must be provided together"
    )

Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

step = 0

if resume_config_path:

    with open(resume_config_path, "r") as f:
        config = json.load(f)

    d_model = config["model"]["d_model"]
    num_heads = config["model"]["num_heads"]
    d_ff = config["model"]["d_ff"]
    vocab_size = config["model"]["vocab_size"]
    context_length = config["model"]["context_length"]
    num_layers = config["model"]["num_layers"]
    theta = config["model"]["theta"]

    max_lr = config["optimizer"]["max_lr"]
    betas = config["optimizer"]["betas"]
    weight_decay = config["optimizer"]["weight_decay"]
    adamw_eps = config["optimizer"]["adamw_eps"]

    min_lr = config["scheduler"]["min_lr"]
    warmup_steps = config["scheduler"]["warmup_steps"]
    cosine_cycle_steps = config["scheduler"]["cosine_cycle_steps"]

    max_grad_norm = config["training"]["max_grad_norm"]
    train_path = config["training"]["train_path"]
    valid_path = config["training"]["valid_path"]
    batch_size = config["training"]["batch_size"]
    total_train_steps = config["training"]["total_train_steps"]
    if args.total_train_steps is not None:
        total_train_steps = args.total_train_steps
        config["training"]["total_train_steps"] = total_train_steps
else:
    d_model = args.d_model
    num_heads = args.num_heads
    d_ff = round(8 / 3 * d_model / 64) * 64
    vocab_size = args.vocab_size
    context_length = args.context_length
    num_layers = args.num_layers
    theta = args.theta

    max_lr = args.max_lr
    betas = args.betas
    weight_decay = args.weight_decay
    adamw_eps = args.adamw_eps

    min_lr = args.min_lr
    warmup_steps = args.warmup_steps
    cosine_cycle_steps = args.cosine_cycle_steps
    max_grad_norm = args.max_grad_norm

    train_path = args.train_path
    valid_path = args.valid_path
    batch_size = args.batch_size
    total_train_steps = args.total_train_steps

    config = {
        "model": {
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": num_layers,
            "theta": theta,
        },
        "optimizer": {
            "max_lr": max_lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "adamw_eps": adamw_eps,
        },
        "scheduler": {
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "cosine_cycle_steps": cosine_cycle_steps,
        },
        "training": {
            "max_grad_norm": max_grad_norm,
            "train_path": train_path,
            "valid_path": valid_path,
            "batch_size": batch_size,
            "total_train_steps": total_train_steps,
        },
    }


with open(f"{checkpoint_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)


model = Transformer_LM(
    d_model,
    num_heads,
    d_ff,
    vocab_size,
    context_length,
    num_layers,
    theta,
    device=device,
    # dtype=dtype,
)

optimizer = AdamW(model.parameters(), max_lr, betas, weight_decay, adamw_eps)

if resume_checkpoint_path:

    step = load_checkpoint(
        resume_checkpoint_path, model, optimizer, map_location=device
    )

train_token_ids = np.load(train_path, mmap_mode="r")
valid_token_ids = np.load(valid_path, mmap_mode="r")

while step < total_train_steps:

    model.train()
    optimizer.zero_grad()

    input_ids, target_ids = get_batch(
        train_token_ids, batch_size, context_length, device
    )

    logits = model(input_ids)

    flat_targets = rearrange(
        target_ids, "batch_size context_length -> (batch_size context_length)"
    )
    flat_logits = rearrange(
        logits,
        "batch_size context_length vocab_size -> (batch_size context_length) vocab_size",
    )

    loss = cross_entropy(flat_logits, flat_targets)

    loss.backward()
    gradient_clipping(model.parameters(), max_grad_norm)

    lr = lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, cosine_cycle_steps)

    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.step()
    completed_steps = step + 1

    if completed_steps % log_interval == 0:
        print(f"completed_steps={completed_steps} cur_lr={lr} train_loss={loss.item()}")

    if completed_steps % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            valid_input_ids, valid_target_ids = get_batch(
                valid_token_ids, batch_size, context_length, device
            )
            valid_logits = model(valid_input_ids)

            valid_flat_targets = rearrange(
                valid_target_ids,
                "batch_size context_length -> (batch_size context_length)",
            )
            valid_flat_logits = rearrange(
                valid_logits,
                "batch_size context_length vocab_size -> (batch_size context_length) vocab_size",
            )
            valid_loss = cross_entropy(valid_flat_logits, valid_flat_targets)

            print(f"completed_steps={completed_steps} valid_loss={valid_loss.item()}")

    if completed_steps % checkpoint_interval == 0:
        checkpoint_path = f"{checkpoint_dir}/ckpt_iter_{completed_steps}.pt"
        latest_path = f"{checkpoint_dir}/latest.pt"
        save_checkpoint(model, optimizer, completed_steps, checkpoint_path)
        save_checkpoint(model, optimizer, completed_steps, latest_path)

    step += 1
