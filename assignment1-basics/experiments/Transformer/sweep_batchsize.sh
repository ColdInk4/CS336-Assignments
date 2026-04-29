GPU=6
TOTAL_TOKENS=40960000
CONTEXT_LENGTH=256
MAX_LR=1.5e-3
MIN_LR=1.5e-4

for BATCH_SIZE in 1 8 16 32 64 128 256 512; do
  STEPS=$((TOTAL_TOKENS / (BATCH_SIZE * CONTEXT_LENGTH)))
  WARMUP_STEPS=$((STEPS / 10))

  echo "Running batch_size=${BATCH_SIZE}, steps=${STEPS}"

  CUDA_VISIBLE_DEVICES=${GPU} uv run python -m cs336_basics.training.training \
    --train_path results/tokenids/ts-train-tokenids.npy \
    --valid_path results/tokenids/ts-valid-tokenids.npy \
    --d_model 512 \
    --vocab_size 10000 \
    --context_length ${CONTEXT_LENGTH} \
    --num_layers 4 \
    --num_heads 16 \
    --theta 10000 \
    --batch_size ${BATCH_SIZE} \
    --total_train_steps ${STEPS} \
    --max_lr ${MAX_LR} \
    --min_lr ${MIN_LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --cosine_cycle_steps ${STEPS} \
    --betas 0.9 0.95 \
    --weight_decay 0.01 \
    --adamw_eps 1e-8 \
    --max_grad_norm 1.0 \
    --log_interval 100 \
    --eval_interval $((STEPS / 10)) \
    --num_eval_batches 20 \
    --checkpoint_interval ${STEPS} \
    --checkpoint_dir checkpoints/tinystories-bs-${BATCH_SIZE} \
    --use_wandb \
    --wandb_project cs336-assignment1 \
    --wandb_run_name ts-bs-${BATCH_SIZE} \
    --device cuda
done
