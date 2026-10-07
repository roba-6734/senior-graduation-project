#!/usr/bin/env bash

set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$HPC_DIR/../.." && pwd)"
HPC_CONFIG="${HPC_CONFIG:-$HPC_DIR/config.env}"

if [[ -f "$HPC_CONFIG" ]]; then
  # The config is a user-owned shell fragment containing paths and scheduler options.
  # shellcheck disable=SC1090
  source "$HPC_CONFIG"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
: "${UNITREE_WORKDIR:?Set UNITREE_WORKDIR in $HPC_CONFIG or export it before running.}"

UNITREE_COMMIT="${UNITREE_COMMIT:-1425b15f73bd4095f0df53709d7c389c3eb9e790}"
UNITREE_REPO="${UNITREE_REPO:-$UNITREE_WORKDIR/unitree_rl_mjlab}"
ENV_PREFIX="${ENV_PREFIX:-$UNITREE_WORKDIR/conda_env}"
MOTION_STEM="${MOTION_STEM:-ayyala_gmr_unitree_g1_29dof_v2_smoothed}"
MOTION_CSV="${MOTION_CSV:-$PROJECT_ROOT/hpc/unitree_mjlab/motions/$MOTION_STEM.csv}"
MOTION_CSV_MANIFEST="${MOTION_CSV_MANIFEST:-$MOTION_CSV.manifest.json}"
MOTION_NPZ="${MOTION_NPZ:-$UNITREE_REPO/src/assets/motions/g1/$MOTION_STEM.npz}"
RETARGET_REPORT="${RETARGET_REPORT:-$PROJECT_ROOT/gmr_output_v2/gmr_v2_evaluation.json}"
STATE_ROOT="${STATE_ROOT:-$UNITREE_WORKDIR/state}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/hpc_results}"
TASK_ID="${TASK_ID:-Unitree-G1-Tracking-No-State-Estimation}"

find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    printf '%s\n' "$CONDA_EXE"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  printf 'Conda was not found. Load your cluster Conda/Miniconda module first.\n' >&2
  return 1
}

activate_unitree_env() {
  local conda_executable conda_base
  conda_executable="$(find_conda)"
  conda_base="$($conda_executable info --base)"
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'Required file does not exist: %s\n' "$path" >&2
    return 1
  fi
}

require_unitree_commit() {
  local actual
  actual="$(git -C "$UNITREE_REPO" rev-parse HEAD)"
  if [[ "$actual" != "$UNITREE_COMMIT" ]]; then
    printf 'Unitree checkout mismatch. Expected %s, found %s\n' "$UNITREE_COMMIT" "$actual" >&2
    return 1
  fi
}

check_gpu() {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the Slurm-assigned GPU")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
PY
}
