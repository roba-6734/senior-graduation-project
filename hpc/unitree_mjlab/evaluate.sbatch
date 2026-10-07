#!/usr/bin/env bash
#SBATCH --job-name=ayyala-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
activate_unitree_env
require_unitree_commit
require_file "$MOTION_NPZ"
check_gpu

SEED="${SEED:-42}"
: "${RUN_TAG:?RUN_TAG must identify the completed training job.}"
STATE_DIR="$STATE_ROOT/$RUN_TAG"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
if [[ -z "$CHECKPOINT_FILE" ]]; then
  require_file "$STATE_DIR/checkpoint.txt"
  CHECKPOINT_FILE="$(<"$STATE_DIR/checkpoint.txt")"
fi
require_file "$CHECKPOINT_FILE"

RESULTS_DIR="${RESULTS_DIR:-$RESULTS_ROOT/$RUN_TAG}"
mkdir -p "$RESULTS_DIR"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

video_args=()
if [[ "${RENDER_VIDEO:-1}" == "1" ]]; then
  video_args=(--video-file "$RESULTS_DIR/ayyala_policy_vs_reference.mp4")
fi
retarget_args=()
if [[ -f "$RETARGET_REPORT" ]]; then
  retarget_args=(--retarget-report "$RETARGET_REPORT")
fi

python "$PROJECT_ROOT/tools/evaluate_mjlab_policy.py" \
  --repo "$UNITREE_REPO" \
  --motion-file "$MOTION_NPZ" \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --output-dir "$RESULTS_DIR" \
  --task "$TASK_ID" \
  --device cuda:0 \
  --seed "$SEED" \
  "${retarget_args[@]}" \
  "${video_args[@]}"

printf 'Evaluation complete: %s\n' "$RESULTS_DIR"
