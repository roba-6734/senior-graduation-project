#!/usr/bin/env bash
#SBATCH --job-name=ayyala-convert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
activate_unitree_env
require_unitree_commit
require_file "$MOTION_CSV"
require_file "$MOTION_CSV_MANIFEST"
check_gpu

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

python "$PROJECT_ROOT/tools/prepare_mjlab_motion.py" validate-csv \
  --input "$MOTION_CSV" \
  --manifest "$MOTION_CSV_MANIFEST"

cd "$UNITREE_REPO"
python scripts/csv_to_npz.py \
  --input-file "$MOTION_CSV" \
  --output-name "$MOTION_STEM.npz" \
  --input-fps 30 \
  --output-fps 50 \
  --robot g1 \
  --device cuda:0

python "$PROJECT_ROOT/tools/prepare_mjlab_motion.py" validate-npz \
  --input "$MOTION_NPZ" \
  --source-manifest "$MOTION_CSV_MANIFEST"
