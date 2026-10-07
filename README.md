# TextMAE-Image-Compression

Learned image compression built on a Masked Autoencoder (MAE). Instead of coding the whole image, the encoder keeps only the most informative patches (chosen from texture and structure scores), codes their features with a hyperprior entropy model, and the decoder fills in the dropped patches with mask tokens.

![demo](https://github.com/tmkhang1999/TextMAE-Image-Compression/assets/74235084/d5354016-76e4-42a5-9da1-4d6a27dfc3c6)

## Results
![result1](./assets/1.png)
![result2](./assets/2.png)

## How it works

```
image --> patch scores (texture x structure)
      \-> patch selection: keep 144 of 196 patches ---------------> ids_restore (Huffman coded)
          MAE encoder (ViT)                                                |
          g_a -> hyperprior (h_a / h_s) -> slice-wise entropy model + LRP  |
          bitstream  ---------------------------------------------------   |
          g_s -> MAE decoder (mask tokens at dropped positions) <----------+
          reconstruction
```

1. **Patch scores** (`textmae/data`): a quad-tree segmentation map (structure) and a Laplacian map (texture) are averaged per 16x16 patch and multiplied. They are computed once by `generate_scores.py`.
2. **Patch selection** (`textmae/models/patch_selection.py`): `stratified` (default, deterministic, spreads the kept patches over score percentiles) or `multinomial` (samples patches proportionally to their score).
3. **Latent coder** (`textmae/models/textmae.py`): 1x1-conv analysis / synthesis transforms, a hyperprior, and a channel-wise autoregressive entropy model with latent residual prediction (compressai building blocks).
4. **Side information**: the order of the patches (`ids_restore`) is Huffman coded (`textmae/coding`) and counted in the reported bpp.
5. **Loss** (`textmae/losses`): `lambda * (0.25 * (1 - SSIM) + 10 * L1 + 0.1 * VGG feature loss) + bpp`.

## Repository layout

```
train.py                 train a model            (see train.sh)
evaluate.py              rate / quality on a test set (see test.sh)
generate_scores.py       precompute patch scores
textmae/
  config.py              command-line options of the three scripts
  data/                  dataset, score maps, per-patch scores
  models/                TextMAE model, patch selection, layer sizes, pos. embeddings
  losses/                rate-distortion loss, VGG perceptual loss
  coding/                Huffman coding of the patch order
  engine/                train / validate loops, evaluation (metrics, real bitstream)
  utils/                 checkpoints and optimizers, logging, distributed helpers
extras/                  optional BLIP-2 captioning and diffusion refiner
tools/                   dataset preparation
notebooks/               Colab quick test
```

## Installation

```bash
pip install -r requirements.txt
```

`timm` must be 0.4.5. `setup.sh` also downloads the official MAE ViT-Large weights into `pretrained_models/`.

## Datasets

| Phase       | Dataset        | Link                                                     |
|-------------|----------------|----------------------------------------------------------|
| Training    | DIV2K          | [Link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)        |
| Training    | Vimeo90K       | [Link](http://toflow.csail.mit.edu/)                     |
| Training    | ImageNet       | [Link](https://image-net.org/download.php)               |
| Testing     | REDS           | [Link](https://seungjunnah.github.io/Datasets/reds.html) |
| Testing     | CLIC           | [Link](https://compression.cc/tasks/#image)              |
| Testing     | Kodak          | [Link](https://r0k.us/graphics/kodak/)                   |

Expected layout (`<name>_scores` is created next to each image folder):

```
datasets/
  train_dataset/
    train/*.png
    val/*.png
  train_dataset_scores/{train,val}.pt
  test_dataset/*.png
  test_dataset_scores/test.pt
```

`tools/prepare_imagenet.py` flattens a Kaggle ImageNet-100 download into `train/` and `val/`.

## Usage

```bash
# 1. patch scores (use the same --input_size for training and evaluation)
python generate_scores.py --training_path datasets/train_dataset --testing_path datasets/test_dataset

# 2. train
python train.py -d datasets/train_dataset --epochs 100 --output_dir weights --log_dir logs

# 3. evaluate (writes reconstructions and report.json to --output_path)
python evaluate.py -d datasets/test_dataset -c weights/best_model.pth -o results --cuda
```

Useful options (`python train.py --help` for all):

| Option | Meaning |
|--------|---------|
| `--model` | `textmae_base_patch16` (default) or `textmae_large_patch16` |
| `--input_size`, `--num_keep_patches` | Image size and number of kept patches (a perfect square, at most `(input_size / 16)^2`) |
| `--patch_selection` | `stratified` (default) or `multinomial` |
| `--pretrained` | Official MAE checkpoint; matching tensors are loaded and the count is printed. The ViT-Large weights fit `textmae_large_patch16` |
| `--lambda` | Rate-distortion trade-off |
| `--resume` | Continue from a `best_model.pth` |

`evaluate.py` reads the architecture settings from the checkpoint; pass `--model`, `--input_size`, `--num_keep_patches` or `--patch_selection` to override them. `--entropy_estimation` estimates the rate from likelihoods instead of coding a real bitstream.

Note: training and evaluation depend on the patch selection and decoder defined in this code. Checkpoints trained with earlier versions of this repository (before the decoder was changed to place the kept tokens at their own positions) are not compatible and should be retrained.

## Contributing

- Keep the model code in `textmae/models` free of script logic; scripts in the repository root only wire configuration, data, model and engine together.
- A new patch selection strategy is a function `(total_scores, num_keep) -> ids_shuffle` registered in `PATCH_SELECTORS`; a new architecture is a preset registered in `MODELS`.
- Layer widths live in `textmae/models/dims.py` (no torch dependency, easy to test).

## Acknowledgements
We appreciate the following repositories for their valuable contributions to our project:
- [MAE](https://github.com/facebookresearch/mae)
- [BLIP](https://github.com/salesforce/BLIP)
- [Stable Diffusion Refiner](https://github.com/Stability-AI/generative-models) (eval)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
