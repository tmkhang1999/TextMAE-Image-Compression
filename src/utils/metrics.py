"""Rate and quality of a trained model, from a real bitstream or from the entropy model's likelihoods."""
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

from models.utils.huffman import huffman_decode, huffman_encode


def psnr(a, b, max_val=255):
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(org, rec, max_val=255):
    """PSNR and MS-SSIM of two [0, 1] image batches after rounding to 8 bit."""
    org, rec = ((t * max_val).clamp(0, max_val).round() for t in (org, rec))
    return {"psnr": psnr(org, rec, max_val).item(), "ms-ssim": ms_ssim(org, rec, data_range=max_val).item()}


def save_output(x_hat, orig_size, file_name, output_dir):
    """Save a reconstruction (1, 3, H, W) resized back to the original (width, height)."""
    img = transforms.ToPILImage()(x_hat.squeeze(0).clamp(0, 1).cpu()).resize(orig_size, Image.BICUBIC)
    img.save(Path(output_dir) / file_name)


@torch.no_grad()
def encode_decode(model, x, total_score):
    """Compress and decompress one image; bpp counts the latent, the hyper-latent and the Huffman-coded patch positions."""
    start = time.time()
    out_enc = model.compress(x, total_score)
    enc_time = time.time() - start

    ids_bits, table = huffman_encode(out_enc["ids_restore"])
    ids_restore = huffman_decode(ids_bits, table, out_enc["ids_restore"].shape, out_enc["ids_restore"].device)

    start = time.time()
    x_hat = model.decompress(out_enc["string"], out_enc["shape"], ids_restore)["x_hat"]
    dec_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    latent_bits = sum(len(s[0]) for s in out_enc["string"]) * 8.0
    return {"x_hat": x_hat, "bpp": (latent_bits + len(ids_bits)) / num_pixels,
            "encoding_time": enc_time, "decoding_time": dec_time}


@torch.no_grad()
def estimate_entropy(model, x, total_score):
    """Forward pass with a likelihood-based rate (no bitstream, no cost for the patch positions)."""
    start = time.time()
    out = model(x, total_score)
    elapsed = time.time() - start
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(l.log().sum() / (-math.log(2) * num_pixels) for l in out["likelihoods"].values())
    # a forward pass cannot separate encoding from decoding, so the time is split evenly
    return {"x_hat": out["x_hat"], "bpp": bpp.item(), "encoding_time": elapsed / 2, "decoding_time": elapsed / 2}


@torch.no_grad()
def evaluate_model(model, dataloader, file_names, output_dir, entropy_estimation=False, half=False):
    """Average PSNR, MS-SSIM, bpp and timings over a test set (batch size 1).

    Metrics are computed against the input resized to the model's input size; the saved reconstructions are
    resized back to the original image size."""
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if half:
        model = model.half()
    run = estimate_entropy if entropy_estimation else encode_decode

    totals = defaultdict(float)
    for index, (img, orig_size, total_score) in enumerate(dataloader):
        img, total_score = img.to(device), total_score.to(device)
        if half:
            img = img.half()
        result = run(model, img, total_score)
        x_hat = result.pop("x_hat").float()
        result.update(compute_metrics(img.float(), x_hat))
        save_output(x_hat, (int(orig_size[0]), int(orig_size[1])), file_names[index], output_dir)
        for k, v in result.items():
            totals[k] += v
    return {k: v / len(file_names) for k, v in totals.items()}
