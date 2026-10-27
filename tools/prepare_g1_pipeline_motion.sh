#!/usr/bin/env bash

set -euo pipefail

# Import an already validated 30 Hz G1 motion into g1-dance-pipeline's
# artifact layout, validate it, and optionally run Isaac Lab preprocessing.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_ROOT="${ROOT}/g1-dance-pipeline"
DEFAULT_INPUT="${ROOT}/hpc/unitree_mjlab/motions/ayyala_gmr_unitree_g1_29dof_v2_smoothed.csv"

usage() {
  cat <<'EOF'
Usage: tools/prepare_g1_pipeline_motion.sh [options]

Options:
  --name NAME       Artifact name (default: ayyala_gmr_v2)
  --input FILE      30 Hz, headerless 36-column G1 CSV
  --input-fps HZ    Input rate (default: 30)
  --output-fps HZ   Training rate (default: 50)
  --no-preprocess   Only import and validate the CSV
EOF
}

name="ayyala_gmr_v2"
input_file="$DEFAULT_INPUT"
input_fps=30
output_fps=50
preprocess=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --input) input_file="$2"; shift 2 ;;
    --input-fps) input_fps="$2"; shift 2 ;;
    --output-fps) output_fps="$2"; shift 2 ;;
    --no-preprocess) preprocess=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || {
  echo "Invalid motion name: $name" >&2
  exit 2
}
[[ -f "$input_file" ]] || { echo "Input motion not found: $input_file" >&2; exit 1; }

job_root="${PIPELINE_ROOT}/artifacts/motions/${name}"
mkdir -p "$job_root"
imported_csv="${job_root}/motion_30hz.csv"
npz_file="${job_root}/motion_50hz.npz"

cp "$input_file" "$imported_csv"
echo "Imported: $imported_csv"

python3 "${PIPELINE_ROOT}/tools/validate_assets.py" \
  --csv "$imported_csv" \
  --csv-fps "$input_fps"

if [[ "$preprocess" -eq 0 ]]; then
  echo "CSV validation complete; preprocessing skipped."
  exit 0
fi

if [[ ! -f "${PIPELINE_ROOT}/config/pipeline.env" ]]; then
  echo "CSV validation complete, but preprocessing is not configured." >&2
  echo "Create ${PIPELINE_ROOT}/config/pipeline.env from pipeline.env.example, then rerun this command." >&2
  exit 3
fi

"${PIPELINE_ROOT}/pipeline.sh" preprocess \
  --name "$name" \
  --input "$imported_csv" \
  --input-fps "$input_fps" \
  --output-fps "$output_fps"

python3 "${PIPELINE_ROOT}/tools/validate_assets.py" --npz "$npz_file"
echo "Prepared training motion: $npz_file"
