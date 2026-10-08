#!/usr/bin/env bash
# Text guidance: BLIP captions the original images, the SDXL refiner restores detail in the decoded ones.
# Downloads about 8 GB of weights on first use; runs on CUDA, Apple MPS or CPU.
#
# Usage: scripts/refine.sh DATASET RECONSTRUCTIONS [OUTPUT_DIR]
#   scripts/refine.sh datasets/kodak results results_refined
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 2 ]; then
    sed -n '2,7p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
python -m src.refine -d "$1" -r "$2" -o "${3:-results_refined}"
