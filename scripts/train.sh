#!/usr/bin/env bash
# Train TextMAE.
#
# Usage: scripts/train.sh DATASET SAVEPATH [MAE_CHECKPOINT]
#   scripts/train.sh datasets/kodak_train weights
#   MODEL=textmae_large_patch16 scripts/train.sh datasets/imagenet weights pretrained_models/mae_visualize_vit_large_ganloss.pth
# MAE_CHECKPOINT initialises the matching weights (the ViT-Large file needs MODEL=textmae_large_patch16).
# Optional env vars: MODEL (textmae_base_patch16), EPOCHS (100), BATCH (16), LAMBDA (1e-4), DEVICE (cuda).
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 2 ]; then
    sed -n '2,8p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
EXTRA=()
if [ -n "${3:-}" ]; then
    [ -f "$3" ] || { echo "error: checkpoint not found: $3" >&2; exit 1; }
    EXTRA+=(--pretrained "$3")
fi

python -m src.training \
    -d "$1" \
    --output_dir "$2" --log_dir "$2/logs" \
    --model "${MODEL:-textmae_base_patch16}" \
    --input_size 224 --num_keep_patches 144 \
    -e "${EPOCHS:-100}" --batch_size "${BATCH:-16}" \
    --lambda "${LAMBDA:-1e-4}" \
    --device "${DEVICE:-cuda}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
