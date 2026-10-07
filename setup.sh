#!/bin/bash
# One-time setup: Python dependencies and the pretrained MAE weights.
set -e

pip install -r requirements.txt

# Official MAE ViT-Large weights (used with --model textmae_large_patch16)
wget -nc -P ./pretrained_models https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

# Optional: ImageNet-100 from Kaggle (needs ~/.kaggle/kaggle.json), flattened to train/ and val/
#   pip install kaggle
#   kaggle datasets download -d ambityga/imagenet100 -p datasets
#   unzip datasets/archive.zip -d datasets/imagenet100
#   python tools/prepare_imagenet.py datasets/imagenet100 datasets/imagenet
