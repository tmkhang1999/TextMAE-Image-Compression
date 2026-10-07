#!/bin/bash
# Evaluate the best checkpoint on the Kodak test images (needs datasets/kodak_scores/test.pt).
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  -d ./datasets/kodak \
  --checkpoint ./weights/best_model.pth \
  --output_path ./results \
  --cuda
