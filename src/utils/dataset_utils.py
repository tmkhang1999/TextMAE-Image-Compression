"""Images with their per-patch importance scores.

    datasets/<name>/{train,val}/*.png   training images        datasets/<name>_scores/{train,val}.pt
    datasets/<name>/*.png               test images            datasets/<name>_scores/test.pt
Scores are made by `python -m src.scores` and stored in the sorted order of `collect_images`.
"""
from pathlib import Path

import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def collect_images(root):
    """Sorted image files below `root` (hidden files such as .DS_Store are skipped)."""
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS and not p.name.startswith("."))


def split_root(dataset_path, mode):
    return Path(dataset_path) if mode == "test" else Path(dataset_path) / mode


def scores_path(dataset_path, mode):
    dataset_path = Path(dataset_path)
    return dataset_path.parent / f"{dataset_path.name}_scores" / f"{mode}.pt"


class ImageScoreDataset(Dataset):
    """Yields (image, original size, patch scores). Train / val are ImageNet-normalised, test stays in [0, 1]."""

    def __init__(self, mode, dataset_path, input_size=224, patch_size=16):
        assert mode in ("train", "val", "test"), f"mode must be train, val or test, got '{mode}'"
        steps = [transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC), transforms.ToTensor()]
        if mode != "test":
            steps.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        self.transform = transforms.Compose(steps)

        self.img_paths = collect_images(split_root(dataset_path, mode))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {split_root(dataset_path, mode)}")
        score_file = scores_path(dataset_path, mode)
        if not score_file.exists():
            raise RuntimeError(f"Score file '{score_file}' does not exist. Run python -m src.scores first.")
        self.scores = torch.load(score_file)
        expected = (len(self.img_paths), (input_size // patch_size) ** 2)
        if tuple(self.scores.shape) != expected:
            raise RuntimeError(f"Score file '{score_file}' has shape {tuple(self.scores.shape)}, expected {expected}. "
                               f"Regenerate it with python -m src.scores --input_size {input_size}.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img), img.size, self.scores[idx]
