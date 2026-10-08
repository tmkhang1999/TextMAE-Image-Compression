#!/usr/bin/env bash
# One-time setup: Python dependencies and the official MAE ViT-Large weights (for --pretrained).
#
# Usage: scripts/setup.sh
# ImageNet-100 (optional, needs Kaggle credentials in ~/.kaggle/kaggle.json):
#   kaggle datasets download -d ambityga/imagenet100 -p datasets && unzip datasets/imagenet100.zip -d datasets/raw
#   python tools/prepare_imagenet.py datasets/raw datasets/imagenet
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -r requirements.txt
wget -nc -P pretrained_models https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
