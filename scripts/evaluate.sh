#!/usr/bin/env bash
# Compress and decompress a test folder with a trained model; prints PSNR, MS-SSIM, bpp and times
# and writes report.json plus the reconstructions.
#
# Usage: scripts/evaluate.sh CHECKPOINT [DATASET] [OUTPUT_DIR]
#   DATASET defaults to datasets/kodak, OUTPUT_DIR to results. Set CUDA=1 to use the GPU.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 1 ]; then
    sed -n '2,7p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
[ -f "$1" ] || { echo "error: checkpoint not found: $1" >&2; exit 1; }

python -m src.inference -c "$1" -d "${2:-datasets/kodak}" -o "${3:-results}" ${CUDA:+--cuda}
