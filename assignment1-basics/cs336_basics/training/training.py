import argparse
from cs336_basics.data_loader import get_batch
import numpy as np
from cs336_basics.transformer import Transformer_LM
from cs336_basics.losses import cross_entropy
from cs336_basics.training import load_checkpoint, save_checkpoint
from cs336_basics.optim import AdamW, gradient_clipping, lr_cosine_schedule
from einops import rearrange
import torch

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
parser.add_argument("--device", help="device")
parser.add_argument("--num_train_steps", type=int, help="num_train_steps")

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
    "--eval_interval", type=int, help="eval interval in steps", default=10
)

parser.add_argument(
    "--checkpoint_interval", type=int, help="checkpoint interval in steps", default=10
)
parser.add_argument("--resume_checkpoint_path", help="checkpoint path in")
parser.add_argument("--checkpoint_dir", help="checkpoint path out", required=True)

# 解析
args = parser.parse_args()

train_path = args.train_path
valid_path = args.valid_path

d_model = args.d_model
vocab_size = args.vocab_size
context_length = args.context_length
num_layers = args.num_layers
num_heads = args.num_heads
theta = args.theta

batch_size = args.batch_size
device = args.device
total_train_steps = args.num_train_steps

max_lr = args.max_lr
min_lr = args.min_lr
warmup_steps = args.warmup_steps
cosine_cycle_steps = args.cosine_cycle_steps
betas = args.betas
weight_decay = args.weight_decay
adamw_eps = args.adamw_eps
max_grad_norm = args.max_grad_norm

eval_interval = args.eval_interval

checkpoint_interval = args.checkpoint_interval

resume_checkpoint_path = args.resume_checkpoint_path
checkpoint_dir = args.checkpoint_dir

train_token_ids = np.load(train_path, mmap_mode="r")
valid_token_ids = np.load(valid_path, mmap_mode="r")

d_ff = round(8 / 3 * d_model / 64) * 64

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

step = 0

if resume_checkpoint_path:
    step = load_checkpoint(resume_checkpoint_path, model, optimizer)


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
        save_checkpoint(model, optimizer, completed_steps, checkpoint_path)

    step += 1
