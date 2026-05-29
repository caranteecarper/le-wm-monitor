#!/usr/bin/env bash
set -euo pipefail

variant="${1:-direct_fastkan}"

if [[ "${variant}" == "baseline_mlp" ]]; then
  pred_proj_type="mlp"
else
  pred_proj_type="${variant}"
fi

python train.py \
  data=dmc \
  seed=3072 \
  num_workers=2 \
  loader.batch_size=8 \
  trainer.precision=16-mixed \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  +trainer.limit_val_batches=1 \
  model.pred_proj_type="${pred_proj_type}" \
  output_model_name="kan_pred_proj_smoke/${variant}/lewm-${variant}" \
  subdir="kan_pred_proj_smoke/${variant}" \
  monitor.enabled=true \
  monitor.csv_logger=true
