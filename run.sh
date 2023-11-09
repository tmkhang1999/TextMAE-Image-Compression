#!/bin/bash

# Clone the repository
git clone https://github.com/tmkhang1999/TextMAE-Image-Compression.git

# Change directory to the cloned repository
cd TextMAE-Image-Compression

# Install the required Python packages
pip install -r requirements.txt

# Training
CUDA_VISIBLE_DEVICES=0 python training.py \
-d ./datasets/imagenet \
--checkpoint ./pretrained_models/mae_visualize_vit_large_ganloss.pth \
--input_size 224 \
--num_keep_patches 144 \
--epochs 50 \
--batch_size 32 \
--output_dir ./weights \
--log_dir ./logs \
--cuda

# Testing
CUDA_VISIBLE_DEVICES=0 python testing.py \
-d ./datasets/kodak \
--checkpoint ./weights/best_model.pth \
--input_size 224 \
--num_keep_patches 144 \
--output_path ./results \
--cuda
