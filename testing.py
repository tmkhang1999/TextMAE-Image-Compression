import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ops import compute_padding
from compressai.zoo import load_state_dict
from pytorch_mssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.Compression.MCM import MCM
from utils.dataloader import get_image_dataset
from utils.huffman import HuffmanCoding

# Constants
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


def collect_images(rootpath: str) -> list[Path | None]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(org: torch.Tensor, rec: torch.Tensor, max_val: int = 255) -> Dict[str, Any]:
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    psnr_value = psnr(org, rec)
    ms_ssim_value = ms_ssim(org, rec, data_range=max_val)
    return {"psnr": psnr_value.item(), "ms-ssim": ms_ssim_value.item()}


def save_output(x, file_name, output_dir):
    x = x.squeeze().clamp(0, 1)
    x = transforms.ToPILImage()(x.cpu())
    x.save(os.path.join(output_dir, file_name))


@torch.no_grad()
def inference(model, x, total_score, file_name, output_dir):
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    ids_dir = os.path.join(output_dir, 'bin')
    os.makedirs(ids_dir, exist_ok=True)

    # Add padding to the input image
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2 ** 6)  # Pad to allow 6 strides of 2
    x_padded = F.pad(x, pad, mode="constant", value=0)

    # Compression process
    start = time.time()
    out_enc = model.compress(x_padded, total_score)
    enc_time = time.time() - start

    # Store the list of IDs for Huffman coding
    ids_keep = out_enc["ids_keep"]
    huffman = HuffmanCoding()
    compressed_ids_keep, device = huffman.compress(ids_keep)
    decompressed_ids_keep = huffman.decompress(compressed_ids_keep, device)

    # Decompression process
    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], decompressed_ids_keep)
    dec_time = time.time() - start

    # Remove padding from the reconstructed image
    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    # Save the reconstructed image
    save_output(out_dec["x_hat"], file_name, output_dir)

    # Calculate metrics for evaluation
    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)

    # Calculate bits per pixel (bpp)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    bpp += (len(compressed_ids_keep) / num_pixels)

    return {
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x):
    start = time.time()
    out = model.forward(x)
    elapsed_time = time.time() - start

    metrics = compute_metrics(x, out["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out["likelihoods"].values())

    return {
        "psnr": metrics["psnr"],
        "ms-ssim": metrics["ms-ssim"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_checkpoint(num_keep_patches: int, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['model'])
    return MCM().from_state_dict(num_keep_patches, state_dict).eval()


def eval_model(
        model: nn.Module,
        output_dir: Path,
        input_dir: Path,
        filepaths: List[Path],
        args,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    metrics = defaultdict(float)

    test_dataset = get_image_dataset(is_train=False,
                                     dataset_path=input_dir,
                                     args=args)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=None,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    for index, (img, ori_shape, total_score) in enumerate(test_dataloader):
        file_name = str(filepaths[index]).split("/")[-1]
        file_name = file_name.split(".")[0]
        if not args.entropy_estimation:
            if args.half:
                model = model.half()
                img = img.to(device).half()
            rv = inference(model, img, total_score, file_name, output_dir)
        else:
            rv = inference_entropy_estimation(model, img)
        for k, v in rv.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    return metrics


def setup_args():
    parser = argparse.ArgumentParser()

    # Common options.
    parser.add_argument("-d", "--dataset", type=str, help="Path to the dataset")
    parser.add_argument("-r", "--recon_path", type=str, default="reconstruction",
                        help="Path to save reconstructed images")

    parser.add_argument("-c", "--entropy-coder",
                        choices=compressai.available_entropy_coders(),
                        default=compressai.available_entropy_coders()[0],
                        help="Entropy coder (default: %(default)s)")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA")
    parser.add_argument("--half", action="store_true", help="Convert model to half floating point (fp16)")
    parser.add_argument("--entropy-estimation", action="store_true",
                        help="Use evaluated entropy estimation (no entropy coding)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-p", "--path", dest="checkpoint_paths", type=str, nargs="*", required=True,
                        help="Checkpoint paths")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")

    # Additional Options
    parser.add_argument("--num_keep_patches", type=int, default=144, required=True,
                        help="Number of patches to keep as input to the model")
    parser.add_argument("--input_size", type=int, default=224, required=True, help="Size of the input image")

    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    runs = args.check_paths
    load_func = load_checkpoint

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write("\rEvaluating {run:s}".format("MCM", run=run))
            sys.stderr.flush()

        model = load_func(args.num_keep_patches, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")

        model.update(force=True)

        metrics = eval_model(
            model,
            args.recon_path,
            args.dataset,
            filepaths,
            args,
        )
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.architecture,
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
