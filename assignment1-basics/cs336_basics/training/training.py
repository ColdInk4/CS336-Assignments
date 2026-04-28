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
import wandb
from time import time
from jaxtyping import Float
from torch import Tensor


def parse_training_args():
    parser = argparse.ArgumentParser(description="Transformer Training")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--train_path", help="path to train tokenids")
    data_group.add_argument("--valid_path", help="path to valid tokenids")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--d_model", type=int, help="d_model")
    model_group.add_argument("--vocab_size", type=int, help="vocab_size")
    model_group.add_argument("--context_length", type=int, help="context_length")
    model_group.add_argument("--num_layers", type=int, help="num_layers")
    model_group.add_argument("--num_heads", type=int, help="num_heads")
    model_group.add_argument("--theta", type=float, help="the ROPE theta")

    optimizer_group = parser.add_argument_group("optimizer")
    optimizer_group.add_argument(
        "--betas", type=float, nargs=2, help="beta, need two nums in adamw"
    )
    optimizer_group.add_argument(
        "--weight_decay", type=float, help="weight decay in adamw"
    )
    optimizer_group.add_argument("--adamw_eps", type=float, help="eps in adamw")

    scheduler_group = parser.add_argument_group("scheduler")
    scheduler_group.add_argument(
        "--max_lr", type=float, help="max_learning_rate in lr_cosine_schedule"
    )
    scheduler_group.add_argument(
        "--min_lr", type=float, help="min_learning_rate in lr_cosine_schedule"
    )
    scheduler_group.add_argument(
        "--warmup_steps", type=int, help="warmup_steps in lr_cosine_schedule"
    )
    scheduler_group.add_argument(
        "--cosine_cycle_steps",
        type=int,
        help="cosine_cycle_steps in lr_cosine_schedule",
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--batch_size", type=int, help="batch_size")
    training_group.add_argument(
        "--total_train_steps", type=int, help="total train steps"
    )
    training_group.add_argument(
        "--max_grad_norm", type=float, help="max_l2_norm in gradient_clipping"
    )
    training_group.add_argument("--device", help="device")

    checkpoint_group = parser.add_argument_group("checkpoint")
    checkpoint_group.add_argument("--resume_config_path", help="config path in")
    checkpoint_group.add_argument("--resume_checkpoint_path", help="checkpoint path in")
    checkpoint_group.add_argument(
        "--checkpoint_dir", help="checkpoint path out", required=True
    )
    checkpoint_group.add_argument(
        "--checkpoint_interval",
        type=int,
        help="checkpoint interval in steps",
        default=10,
    )

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument(
        "--log_interval", type=int, help="log interval in steps", default=10
    )
    logging_group.add_argument(
        "--eval_interval", type=int, help="eval interval in steps", default=10
    )
    logging_group.add_argument(
        "--num_eval_batches", type=int, help="eval batches", default=10
    )
    logging_group.add_argument("--use_wandb", action="store_true")
    logging_group.add_argument("--wandb_project", default="cs336-assignment1")
    logging_group.add_argument("--wandb_run_name")

    return parser.parse_args()


def compute_lm_loss(logits, target_ids) -> Float[Tensor, ""]:
    logits = rearrange(
        logits,
        "batch_size context_length vocab_size -> (batch_size context_length) vocab_size",
    )
    target_ids = rearrange(
        target_ids,
        "batch_size context_length -> (batch_size context_length)",
    )
    return cross_entropy(logits, target_ids)


def evaluate_and_log(
    model,
    run,
    completed_steps,
    start_time,
    num_eval_batches,
    valid_token_ids,
    batch_size,
    context_length,
    device,
):
    model.eval()
    with torch.no_grad():
        loss = 0
        for _ in range(num_eval_batches):
            input_ids, target_ids = get_batch(
                valid_token_ids, batch_size, context_length, device
            )
            logits = model(input_ids)

            loss += compute_lm_loss(logits, target_ids).item()
        loss /= num_eval_batches
        elapsed_sec = time() - start_time

        if run is not None:
            wandb.log(
                {
                    "valid/loss": loss,
                    "time/elapsed_sec": elapsed_sec,
                },
                step=completed_steps,
            )
        else:
            print(
                f"completed_steps={completed_steps} valid_loss={loss} elapsed_sec={elapsed_sec}"
            )


def main():

    args = parse_training_args()

    device = args.device

    log_interval = args.log_interval
    eval_interval = args.eval_interval
    num_eval_batches = args.num_eval_batches

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
        config["runtime"]["log_interval"] = log_interval
        config["runtime"]["eval_interval"] = eval_interval
        config["runtime"]["num_eval_batches"] = num_eval_batches
        config["runtime"]["checkpoint_interval"] = checkpoint_interval
        config["runtime"]["device"] = device
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
            "runtime": {
                "log_interval": log_interval,
                "eval_interval": eval_interval,
                "num_eval_batches": num_eval_batches,
                "checkpoint_interval": checkpoint_interval,
                "device": device,
            },
        }

    run = None
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
        )
    with open(Path(checkpoint_dir) / "config.json", "w") as f:
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
    start_time = time()
    start_step = step
    evaluate_and_log(
        model,
        run,
        step,
        start_time,
        num_eval_batches,
        valid_token_ids,
        batch_size,
        context_length,
        device,
    )
    while step < total_train_steps:

        model.train()
        optimizer.zero_grad()

        input_ids, target_ids = get_batch(
            train_token_ids, batch_size, context_length, device
        )

        logits = model(input_ids)

        loss = compute_lm_loss(logits, target_ids)

        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)

        lr = lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, cosine_cycle_steps)

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()
        completed_steps = step + 1

        if completed_steps % log_interval == 0:
            elapsed_sec = time() - start_time
            run_tokens_processed = (
                (completed_steps - start_step) * batch_size * context_length
            )
            tokens_per_sec = run_tokens_processed / elapsed_sec
            if run is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "time/elapsed_sec": elapsed_sec,
                        "train/run_tokens_processed": run_tokens_processed,
                        "train/tokens_per_sec": tokens_per_sec,
                    },
                    step=completed_steps,
                )
            else:
                print(
                    f"completed_steps={completed_steps} cur_lr={lr} train_loss={loss.item()} elapsed_sec={elapsed_sec} tokens_processed={run_tokens_processed} tokens_per_sec={tokens_per_sec}"
                )

        if completed_steps % eval_interval == 0:

            evaluate_and_log(
                model,
                run,
                completed_steps,
                start_time,
                num_eval_batches,
                valid_token_ids,
                batch_size,
                context_length,
                device,
            )

        if completed_steps % checkpoint_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/ckpt_iter_{completed_steps}.pt"
            latest_path = f"{checkpoint_dir}/latest.pt"
            save_checkpoint(model, optimizer, completed_steps, checkpoint_path)
            save_checkpoint(model, optimizer, completed_steps, latest_path)

        step += 1

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
