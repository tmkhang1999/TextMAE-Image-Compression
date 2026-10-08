#!/usr/bin/env bash
# Precompute the per-patch importance scores (needed once per dataset and input size).
#
# Usage: scripts/scores.sh TRAIN_DIR TEST_DIR
#   scripts/scores.sh datasets/kodak_train datasets/kodak_test
# TRAIN_DIR holds train/ and val/; scores go to <dir>_scores/. Optional env var: INPUT_SIZE (default 224).
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 2 ]; then
    sed -n '2,6p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
python -m src.scores --training_path "$1" --testing_path "$2" --input_size "${INPUT_SIZE:-224}"
