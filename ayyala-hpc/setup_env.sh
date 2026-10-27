#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 ]]; then
  export HPC_CONFIG="$1"
fi
config_to_use="${HPC_CONFIG:-$SCRIPT_DIR/config.env}"
if [[ ! -f "$config_to_use" ]]; then
  printf 'Create %s from config.env.example before setup.\n' "$config_to_use" >&2
  exit 1
fi
export HPC_CONFIG="$(realpath "$config_to_use")"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

mkdir -p "$UNITREE_WORKDIR"

if [[ ! -d "$UNITREE_REPO/.git" ]]; then
  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git "$UNITREE_REPO"
  git -C "$UNITREE_REPO" checkout --detach "$UNITREE_COMMIT"
else
  require_unitree_commit
fi

conda_executable="$(find_conda)"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$conda_executable" create --yes --prefix "$ENV_PREFIX" --channel conda-forge \
    python=3.11 pip cmake ninja yaml-cpp boost-cpp eigen spdlog fmt
fi

"$conda_executable" run --prefix "$ENV_PREFIX" python -m pip install --upgrade pip
"$conda_executable" run --prefix "$ENV_PREFIX" python -m pip install --editable "$UNITREE_REPO"
"$conda_executable" run --prefix "$ENV_PREFIX" python -m pip freeze \
  > "$UNITREE_WORKDIR/environment-pip-freeze.txt"
"$conda_executable" list --prefix "$ENV_PREFIX" --explicit \
  > "$UNITREE_WORKDIR/environment-conda-explicit.txt"

"$conda_executable" run --prefix "$ENV_PREFIX" python - <<'PY'
import platform
import torch
import mjlab

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_runtime={torch.version.cuda}")
print(f"mjlab={mjlab.__file__}")
print(f"cuda_visible_now={torch.cuda.is_available()} (a login node may correctly report False)")
PY

printf 'Environment ready at %s\nUnitree checkout: %s\n' "$ENV_PREFIX" "$UNITREE_REPO"
