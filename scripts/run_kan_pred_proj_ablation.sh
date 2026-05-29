#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating the LeWM environment.
# This script trains all pred_proj variants from scratch with identical settings.

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SEED="${SEED:-3072}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DATA_CONFIG="${DATA_CONFIG:-dmc}"
RUN_ROOT="${RUN_ROOT:-kan_pred_proj_ablation}"
PIN_MEMORY="${PIN_MEMORY:-false}"

variants=(
  baseline_mlp
  direct_fastkan
  bottleneck_mlp_control
  bottleneck_fastkan
  sparse_bottleneck_fastkan
)

for variant in "${variants[@]}"; do
  case "${variant}" in
    baseline_mlp)
      pred_proj_type="mlp"
      ;;
    *)
      pred_proj_type="${variant}"
      ;;
  esac

  run_name="${RUN_ROOT}/${variant}_seed${SEED}"
  model_name="${RUN_ROOT}/${variant}_seed${SEED}/lewm-${variant}"

  echo "================================================================================"
  echo "variant=${variant}"
  echo "pred_proj_type=${pred_proj_type}"
  echo "run_name=${run_name}"
  date

  python train.py \
    data="${DATA_CONFIG}" \
    seed="${SEED}" \
    num_workers="${NUM_WORKERS}" \
    loader.batch_size="${BATCH_SIZE}" \
    loader.pin_memory="${PIN_MEMORY}" \
    trainer.precision=16-mixed \
    trainer.max_epochs="${MAX_EPOCHS}" \
    model.pred_proj_type="${pred_proj_type}" \
    output_model_name="${model_name}" \
    subdir="${run_name}" \
    monitor.enabled=true \
    monitor.csv_logger=true

  date
done
