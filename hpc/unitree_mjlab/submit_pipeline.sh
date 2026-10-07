#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"
if [[ $# -gt 1 ]]; then
  export HPC_CONFIG="$2"
fi
config_to_use="${HPC_CONFIG:-$SCRIPT_DIR/config.env}"
if [[ ! -f "$config_to_use" ]]; then
  printf 'Create %s from config.env.example before submitting.\n' "$config_to_use" >&2
  exit 1
fi
export HPC_CONFIG="$(realpath "$config_to_use")"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_file "$MOTION_CSV"

case "$MODE" in
  smoke)
    NUM_ENVS="${SMOKE_NUM_ENVS:-512}"
    MAX_ITERATIONS="${SMOKE_MAX_ITERATIONS:-50}"
    ;;
  full)
    NUM_ENVS="${NUM_ENVS:-4096}"
    MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
    ;;
  *)
    printf 'Usage: %s {smoke|full} [config.env]\n' "$0" >&2
    exit 2
    ;;
esac

SEED="${SEED:-42}"
RUN_TAG="ayyala_v2_${MODE}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${SLURM_LOG_DIR:-$UNITREE_WORKDIR/slurm_logs}"
mkdir -p "$LOG_DIR"

sbatch_options=(--parsable --output="$LOG_DIR/%x-%j.out")
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  sbatch_options+=(--partition="$SLURM_PARTITION")
fi
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  sbatch_options+=(--account="$SLURM_ACCOUNT")
fi
if [[ -n "${SLURM_GRES:-}" ]]; then
  sbatch_options+=(--gres="$SLURM_GRES")
fi

export_values="ALL,HPC_CONFIG=$HPC_CONFIG,RUN_TAG=$RUN_TAG,SEED=$SEED,NUM_ENVS=$NUM_ENVS,MAX_ITERATIONS=$MAX_ITERATIONS"
convert_job="$(sbatch "${sbatch_options[@]}" --export="$export_values" "$SCRIPT_DIR/prepare_motion.sbatch")"
convert_job_id="${convert_job%%;*}"
train_job="$(sbatch "${sbatch_options[@]}" --dependency="afterok:$convert_job_id" --export="$export_values" "$SCRIPT_DIR/train.sbatch")"
train_job_id="${train_job%%;*}"
eval_job="$(sbatch "${sbatch_options[@]}" --dependency="afterok:$train_job_id" --export="$export_values" "$SCRIPT_DIR/evaluate.sbatch")"

printf 'Submitted %s pipeline\n' "$MODE"
printf '  run tag: %s\n' "$RUN_TAG"
printf '  convert job: %s\n' "$convert_job"
printf '  train job: %s\n' "$train_job"
printf '  evaluate job: %s\n' "$eval_job"
printf '  results: %s/%s\n' "$RESULTS_ROOT" "$RUN_TAG"
