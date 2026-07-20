#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/com-FT.ckpt}"
CCD_PATH="${CCD_PATH:-$ROOT/checkpoints/ccd.pkl}"
GPU_ID="${GPU_ID:-0}"

cd "$ROOT"

highss \
  --target_pdb "$ROOT/examples/mdm2/mdm2_pro.pdb" \
  --target_chains A \
  --target_name mdm2 \
  --binder_length 25 \
  --disulfide_pairs "2-25,6-21,10-16" \
  --msa_mode target \
  --msa_max_seqs 4096 \
  --checkpoint "$CHECKPOINT" \
  --ccd_path "$CCD_PATH" \
  --gpu_id "$GPU_ID" \
  --design_samples 32 \
  --suffix mdm2_25aa_3ss_target_msa
