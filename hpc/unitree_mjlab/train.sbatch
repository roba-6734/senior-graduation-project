#!/usr/bin/env bash
#SBATCH --job-name=ayyala-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
activate_unitree_env
require_unitree_commit
require_file "$MOTION_NPZ"
check_gpu

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-ayyala_v2_seed${SEED}_${SLURM_JOB_ID:-manual}}"
STATE_DIR="$STATE_ROOT/$RUN_TAG"
mkdir -p "$STATE_DIR"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

cd "$UNITREE_REPO"
python scripts/train.py "$TASK_ID" \
  --motion_file="$MOTION_NPZ" \
  --env.scene.num-envs="$NUM_ENVS" \
  --agent.max-iterations="$MAX_ITERATIONS" \
  --agent.seed="$SEED" \
  --agent.run-name="$RUN_TAG" \
  --enable-nan-guard=True

run_dir="$(find logs/rsl_rl/g1_tracking -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_TAG}" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$run_dir" ]]; then
  printf 'Could not locate the training directory for run tag %s\n' "$RUN_TAG" >&2
  exit 1
fi
checkpoint="$(find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -print | sort -V | tail -n 1)"
require_file "$checkpoint"

printf '%s\n' "$(realpath "$run_dir")" > "$STATE_DIR/run_dir.txt"
printf '%s\n' "$(realpath "$checkpoint")" > "$STATE_DIR/checkpoint.txt"
printf '%s\n' "$RUN_TAG" > "$STATE_DIR/run_tag.txt"
printf 'Training complete. Checkpoint: %s\n' "$checkpoint"
