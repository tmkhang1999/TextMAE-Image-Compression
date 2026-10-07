#!/bin/bash
# Train on the Kodak split. Scores must exist first: see generate_scores.py / README.
#
# --pretrained takes an official MAE checkpoint. The ViT-Large weights below only fit
# --model textmae_large_patch16; with the default base model only the matching decoder
# tensors are loaded (train.py prints how many).

# ImageNet run:
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   -d ./datasets/imagenet \
#   --pretrained ./pretrained_models/mae_visualize_vit_large_ganloss.pth \
#   --input_size 224 \
#   --num_keep_patches 144 \
#   --epochs 1000 \
#   --batch_size 32 \
#   --output_dir ./weights \
#   --log_dir ./logs

CUDA_VISIBLE_DEVICES=0 python train.py \
  -d ./datasets/kodak_train \
  --pretrained ./pretrained_models/mae_visualize_vit_large_ganloss.pth \
  --input_size 224 \
  --num_keep_patches 144 \
  --epochs 10 \
  --batch_size 4 \
  --output_dir ./weights \
  --log_dir ./logs
