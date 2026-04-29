TOTAL_TOKENS=40960000
CONTEXT_LENGTH=256
BATCH_SIZE=128
LR=2e-3
GPU=1

STEPS=$((TOTAL_TOKENS / (BATCH_SIZE * CONTEXT_LENGTH)))
WARMUP_STEPS=$((STEPS / 10))
EVAL_INTERVAL=$((STEPS / 10))
LOG_INTERVAL=$((STEPS / 20))
CHECKPOINT_INTERVAL=$((STEPS / 5))

if [ "$EVAL_INTERVAL" -lt 1 ]; then EVAL_INTERVAL=1; fi
if [ "$LOG_INTERVAL" -lt 1 ]; then LOG_INTERVAL=1; fi
if [ "$CHECKPOINT_INTERVAL" -lt 1 ]; then CHECKPOINT_INTERVAL=1; fi

CUDA_VISIBLE_DEVICES="$GPU" uv run python -m cs336_basics.training.training \
  --train_path results/tokenids/ts-train-tokenids.npy \
  --valid_path results/tokenids/ts-valid-tokenids.npy \
  --d_model 512 \
  --vocab_size 10000 \
  --context_length 256 \
  --num_layers 4 \
  --num_heads 16 \
  --theta 10000 \
  --batch_size "$BATCH_SIZE" \
  --total_train_steps "$STEPS" \
  --max_lr "$LR" \
  --min_lr "$(python - <<PY
print(float("$LR") * 0.1)
PY
)" \
  --warmup_steps "$WARMUP_STEPS" \
  --cosine_cycle_steps "$STEPS" \
  --betas 0.9 0.95 \
  --weight_decay 0.01 \
  --adamw_eps 1e-8 \
  --max_grad_norm 1.0 \
  --log_interval "$LOG_INTERVAL" \
  --eval_interval "$EVAL_INTERVAL" \
  --num_eval_batches 10 \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --checkpoint_dir "checkpoints/tinystories-bs128-lr-$LR-40960000" \
  --use_wandb \
  --wandb_project cs336-assignment1 \
  --wandb_run_name "ts-bs128-lr-$LR-40960000" \
  --device cuda
