<div align="center">

# TextMAE: Text-Guided Masked Autoencoders for Low-Bitrate Image Compression

**Send the patches that matter and one sentence of text. A Masked Autoencoder and a diffusion model rebuild the rest.**

[Khang Tran](https://github.com/tmkhang1999) &nbsp;|&nbsp; personal research project, 2023

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C)
![CompressAI](https://img.shields.io/badge/CompressAI-1.2.4-4C72B0)
![License](https://img.shields.io/badge/License-Unlicense-lightgrey)

<img src="assets/2.png" width="85%" alt="Masked inputs and reconstructions on Kodak">

<sub>Left: what the encoder sees when 75, 50 or 25 % of the patches are dropped (gray). Middle: original. Right: decoded images at three bit rates.</sub>

</div>

## Overview

Most image codecs spend the same effort on every region, including smooth sky and blurred background that a neural network can fill in by itself. The purpose of this project is to test whether a codec can send only the informative patches and still reconstruct a good image at very low bit rates. TextMAE does this in three steps: it drops the easy patches, codes the remaining ones with a learned codec, and adds a one-sentence caption so that a diffusion model can restore detail.

On two Kodak images, TextMAE reaches **22 dB at a reported 0.02 bpp**, a rate at which JPEG and WebP cannot produce an image at all (see the note on patch positions in the [Discussion](#discussion)). At 0.12 bpp it is 6.5 dB better than JPEG at a similar size ([Results](#results)).

<p align="center">
  <img src="assets/figures/overview.svg" width="100%" alt="TextMAE overview: select patches, code them, decode, refine with a caption">
</p>

Four terms are used throughout. **bpp** (bits per pixel) is the file size divided by the number of pixels, so lower means a smaller file. **PSNR** (in dB) measures how close the decoded image is to the original, so higher means a more faithful image. **MAE** (Masked Autoencoder) is a Vision Transformer trained to rebuild an image from a few visible patches; here it encodes the kept patches and fills in the dropped ones. **LIC** (learned image compression) is the part that turns features into a small bitstream: transforms, a quantizer and an entropy coder.

## Method

The figure above shows the three stages. Light blue boxes are the MAE backbone and dark blue boxes the LIC part, both trained here. Orange boxes are handcrafted or entropy coding, and white boxes are frozen pretrained models. The bit rate on each arrow is the cost of that stage.

**First, patch selection.** Every 16x16 patch receives a score: the product of a *structure* score (quad-tree segmentation) and a *texture* score (absolute Laplacian), normalised to [0, 1]. Smooth patches score low because the decoder can infer them from their neighbours. The default *percentile sampling* keeps 144 of 196 patches: it always keeps the top score bucket and shares the rest of the budget across the other buckets by the softmax of their mean score. A *multinomial* variant samples patches in proportion to their score.

<p align="center">
  <img src="assets/figures/patch_selection.png" width="100%" alt="Patch scores and selection on kodim23">
</p>
<p align="center"><sub>Scoring and selection on Kodak <code>kodim23</code>. Gray patches are dropped before encoding; the eyes, beaks and feather edges are kept.</sub></p>

**Second, the learned codec.** It has two parts. The MAE backbone (ViT-B) encodes only the kept patches and, at the end, predicts all patches from them. The LIC part compresses the features: $g_a$ maps the 12 x 12 token grid to the latent $y$, a hyperprior and a channel-wise context model predict a Gaussian for each of 12 slices of $y$, and these predictions drive an rANS arithmetic coder. Because the decoder must know where the kept patches belong, their positions are sent as Huffman-coded side information. The decoder inserts a learned mask token at each of the 52 dropped positions and predicts all 196 patches. The codec is trained end to end with a rate-distortion loss, where $\mathcal{L}_{\text{VGG}}$ compares VGG-16 features and $\lambda$ balances size against quality:

$$
\mathcal{L} = \underbrace{\mathbb{E}\left[-\log_2 p(\hat y \mid \hat z) - \log_2 p(\hat z)\right] / N_{\text{pixels}}}_{\text{rate (bpp)}} + \lambda \left(0.25\,(1-\mathrm{SSIM}) + 10\,\lVert x-\hat x\rVert_1 + 0.1\,\mathcal{L}_{\text{VGG}}\right)
$$

<p align="center">
  <img src="assets/figures/codec.svg" width="100%" alt="MAE codec and entropy model">
</p>
<p align="center"><sub>Detail of the learned codec. (a) The MAE (ViT) encoder and the LIC transform <i>g<sub>a</sub></i> map the kept patches to the latent <i>y</i>, which is quantized (Q) and arithmetic-coded (AE / AD); <i>g<sub>s</sub></i> and the MAE decoder reconstruct the image, inserting mask tokens at the dropped positions. (b) The LIC entropy model predicts a Gaussian (&mu;, &sigma;) for every latent.</sub></p>

**Third, text guidance.** BLIP describes the original image in one sentence, and the caption is sent with the bitstream as plain text. At the decoder, the Stable Diffusion XL refiner runs image-to-image on $\hat x$ with the caption as its prompt, so the detail it adds matches what the image shows. The reported rate is the bits of $y$, $z$, the patch positions and the caption, divided by the number of pixels.

<p align="center">
  <img src="assets/figures/refinement.jpg" width="85%" alt="Decoded vs caption-guided refinement">
</p>
<p align="center"><sub>A real run of this stage on an Apple M3 Pro (<code>tools/refine_example.py</code>). The caption <i>"there are two parrots that are standing next to each other"</i> costs 58 bytes (+0.009 bpp). The refiner restores feather texture and a sharp eye, but it redraws the facial stripes differently from the original.</sub></p>

## Results

<p align="center">
  <img src="assets/figures/rd_kodak.png" width="85%" alt="Rate-distortion on two Kodak images">
</p>

Each point is one compressed file, and up and to the left is better. JPEG and WebP were measured with Pillow on the same 224 x 224 inputs, using the PSNR of `evaluate.py`. The TextMAE points come from the trained model of the original experiments and are measured before refinement.

| Image | Low rate | Medium rate | High rate |
|:------|:------:|:------:|:------:|
| Aircraft (kodim20) | 0.020 bpp / 22.44 dB | 0.07 bpp / 25.6 dB | 0.15 bpp / 27.8 dB |
| Parrots (kodim23)  | 0.018 bpp / 22.2 dB  | 0.06 bpp / 26.1 dB | 0.12 bpp / 27.5 dB |

1. **Below 0.13 bpp, only TextMAE works, if its rates are right.** JPEG stops at about 0.13 bpp and WebP at about 0.15 to 0.18 bpp, while TextMAE still gives 22 dB at 0.02 bpp.
2. **At similar sizes, TextMAE is clearly better than JPEG.** On the parrots, it gives 27.5 dB at 0.12 bpp, against 21.0 dB for JPEG at 0.14 bpp.
3. **Against WebP, TextMAE is on par or better.** On the aircraft, the two are within 0.3 dB at 0.15 bpp (27.8 against 27.5 dB). On the parrots, TextMAE reaches 27.5 dB at 0.12 bpp, while WebP's lowest setting gives 26.6 dB at 0.18 bpp.

<p align="center">
  <img src="assets/1.png" width="85%" alt="Kodak aircraft">
</p>

## Discussion

These results are encouraging, but they should be read with care.

- **The evaluation is small.** It covers two Kodak images at three rates, and no run without patch dropping was done, so the contribution of each idea is not measured.
- **Generated detail is plausible, not faithful.** The refiner invents texture that fits the caption (see the facial stripes above). It can look better while scoring lower in PSNR, so a perceptual metric such as LPIPS or FID would judge it more fairly.
- **Patch positions are expensive, and the reported rates may not include them.** The decoder needs to know where the kept patches belong. The positions are Huffman-coded, but the 196 indices are all distinct, so the coder saves nothing: they cost 1,508 bits, or 0.030 bpp at 224 x 224, more than the 0.02 bpp point above. The original experiments' rates could not be re-checked, so they may leave this cost out. Sorting the kept tokens by index would reduce it to a 196-bit mask (about 0.003 bpp), at the price of retraining. The 58-byte caption adds a further 0.009 bpp.
- **The decoder is heavy.** BLIP and SDXL need about 8 GB of weights, and refining one 1024 x 1024 image takes about a minute on an M3 Pro (64 and 82 s in two runs). This suits archival storage more than real-time use.
- **The code changed after the experiments.** Three bugs were fixed: the decoder is now placed correctly (each kept token was one position off), the texture map uses the original image, and ties between equal scores are broken at random. The old checkpoints are therefore not compatible and would need retraining.

In sum, dropping patches by score and guiding the decoder with a short caption is a promising way to reach rates below those of JPEG and WebP. A full Kodak evaluation and an ablation are the natural next steps.

## Related work

TextMAE builds on the hyperprior and channel-wise entropy models of Ball&eacute; et al. (2018) and Minnen & Singh (2020), and on the Masked Autoencoder of He et al. (2022). Text-guided compression at ultra-low bit rates appeared in parallel: Text + Sketch (Lei et al., 2023) and PerCo (Careil et al., ICLR 2024) send a caption and regenerate the image with a diffusion model. TextMAE differs in that it keeps a conventional codec for the important patches and uses the caption only to refine the result, so the output stays anchored to the transmitted pixels.

## Getting started

```bash
git clone https://github.com/tmkhang1999/TextMAE-Image-Compression.git && cd TextMAE-Image-Compression
bash scripts/setup.sh                                                    # dependencies (timm 0.4.5) and MAE weights

bash scripts/scores.sh datasets/kodak_train datasets/kodak_test         # patch scores, once per dataset
bash scripts/train.sh datasets/kodak_train weights                      # train the codec
bash scripts/evaluate.sh weights/best_model.pth datasets/kodak results  # real bitstream: PSNR, MS-SSIM, bpp
bash scripts/refine.sh datasets/kodak results results_refined          # BLIP caption + SDXL (CUDA, MPS or CPU)
```

Every script prints its options when run without arguments, and `python -m src.training --help` lists all training options (`--model`, `--num_keep_patches`, `--patch_selection`, `--pretrained`, `--lambda`). Training expects `datasets/<name>/{train,val}/*.png` and testing `datasets/<name>/*.png`; the scores go to `datasets/<name>_scores/`, and the Kodak images are included. `python -m tools.figures.make_figures` rebuilds the figures of this README.

```
models/    TextMAE (MAE + LIC), patch selection, text guidance (BLIP, SDXL)
src/       training.py, inference.py, refine.py, scores.py, losses/, utils/
scripts/   train.sh, evaluate.sh, refine.sh, scores.sh, setup.sh
tools/     figures/ (README figures), prepare_imagenet.py
```

## Acknowledgements and citation

This project builds on [MAE](https://github.com/facebookresearch/mae), [CompressAI](https://github.com/InterDigitalInc/CompressAI), [BLIP](https://github.com/salesforce/BLIP) and the [Stable Diffusion XL refiner](https://github.com/Stability-AI/generative-models).

```bibtex
@misc{tran2023textmae,
  author       = {Khang Tran},
  title        = {TextMAE: Text-Guided Masked Autoencoders for Low-Bitrate Image Compression},
  year         = {2023},
  howpublished = {\url{https://github.com/tmkhang1999/TextMAE-Image-Compression}}
}
```
