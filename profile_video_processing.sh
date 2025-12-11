#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------
# Config
# -----------------------------------------

# Path to SeBS repo (adjust if you put this in a different repo)
SEBS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Nsight Compute output file
NCU_OUT="${SEBS_ROOT}/profiling/vp_ncu_raw.csv"

# Metrics to collect (edit if you want more/less)
METRICS="\
smsp__warp_issue_stalled_memory_dependency,\
sm__pipe_fp32_cycles_active,\
l1tex__t_sectors_miss,\
lts__t_sectors_miss,\
dram__bytes_read.sum"

# -----------------------------------------
# Sanity checks
# -----------------------------------------

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: Nsight Compute CLI 'ncu' not found in PATH." >&2
  echo "Make sure Nsight Compute is installed and ncu is in your PATH." >&2
  exit 1
fi

if [ ! -d "${SEBS_ROOT}/python-venv" ]; then
  echo "ERROR: python-venv not found in ${SEBS_ROOT}." >&2
  echo "Run './install.py --local' in the SeBS repo root first." >&2
  exit 1
fi

# -----------------------------------------
# Activate SeBS virtualenv
# -----------------------------------------

# shellcheck source=/dev/null
source "${SEBS_ROOT}/python-venv/bin/activate"

# -----------------------------------------
# Command that runs your video_processing benchmark locally
# -----------------------------------------
# NOTE: if your actual command differs, just edit CMD below.
# This is the only line you may need to tweak.

CMD="python ${SEBS_ROOT}/sebs.py \
  --config ${SEBS_ROOT}/config/local.json \
  benchmark run \
  --benchmark video-processing \
  --size small \
  --platform local"

echo "Running Nsight Compute on video_processing benchmark..."
echo "Metrics: ${METRICS}"
echo "Output CSV: ${NCU_OUT}"
echo

# -----------------------------------------
# Run with Nsight Compute
# -----------------------------------------

ncu \
  --metrics "${METRICS}" \
  --csv \
  --log-file "${NCU_OUT}" \
  --target-processes all \
  --set default \
  ${CMD}

echo
echo "Done. Raw Nsight CSV saved to:"
echo "  ${NCU_OUT}"
echo
echo "Next step: parse it with parse_vp_ncu_csv.py"
