#!/usr/bin/env bash
# Run one MDM2 design in the selected disulfide-loss mode.
# Usage: bash examples/run_highss_disulfide_modes.sh [on|off]
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: bash $0 [on|off]" >&2
  exit 2
fi

MODE="${1:-on}"
MODE_ARGS=()
case "$MODE" in
  off)
    MODE_ARGS=(--use_disulfide_loss false)
    ;;
  on)
    MODE_ARGS=(--use_disulfide_loss true)
    ;;
  *)
    echo "Unknown mode: $MODE. Choose on or off." >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/com-FT.ckpt}"
CCD_PATH="${CCD_PATH:-$ROOT/checkpoints/ccd.pkl}"
GPU_ID="${GPU_ID:-0}"
MSA_MODE="${MSA_MODE:-target}"
DESIGN_SAMPLES="${DESIGN_SAMPLES:-1}"
SEED="${SEED:-1}"
WORK_DIR="${WORK_DIR:-$ROOT}"
SUFFIX="${SUFFIX:-mdm2_${MODE}_seed${SEED}}"

cd "$ROOT"
exec highss \
  --target_pdb "$ROOT/examples/mdm2/mdm2_pro.pdb" \
  --target_chains A \
  --target_name mdm2 \
  --binder_length 18 \
  --disulfide_pairs "2-17,6-13" \
  --msa_mode "$MSA_MODE" \
  --checkpoint "$CHECKPOINT" \
  --ccd_path "$CCD_PATH" \
  --gpu_id "$GPU_ID" \
  --design_samples "$DESIGN_SAMPLES" \
  --seed "$SEED" \
  --work_dir "$WORK_DIR" \
  --suffix "$SUFFIX" \
  "${MODE_ARGS[@]}"
