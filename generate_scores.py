"""
Precompute the per-patch importance scores used by the dataset.

Writes <dataset>_scores/{train,val,test}.pt next to each dataset folder, one row of
(input_size / patch_size) ** 2 scores per image, in `collect_images` order.
"""
from pathlib import Path

import torch
from tqdm import tqdm

from textmae.config import generate_scores_parser
from textmae.data.dataset import collect_images, scores_path, split_root
from textmae.data.patch_scores import image_patch_scores, read_gray


def generate(mode, dataset_path, input_size, patch_size):
    image_paths = collect_images(split_root(dataset_path, mode))
    if not image_paths:
        raise RuntimeError(f"No images found in {split_root(dataset_path, mode)}")

    scores = torch.stack([
        torch.from_numpy(image_patch_scores(read_gray(path), input_size, patch_size))
        for path in tqdm(image_paths, desc=f"{Path(dataset_path).name}/{mode}")
    ])
    print(f"{mode}: scores of shape {tuple(scores.shape)}")

    out_file = scores_path(dataset_path, mode)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, out_file)
    print(f"Saved {out_file}")


def main(args):
    for mode in ("train", "val"):
        generate(mode, args.training_path, args.input_size, args.patch_size)
    generate("test", args.testing_path, args.input_size, args.patch_size)


if __name__ == "__main__":
    main(generate_scores_parser().parse_args())
