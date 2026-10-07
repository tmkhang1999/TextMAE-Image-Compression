"""
Image dataset that yields (image, original size, per-patch importance scores).

Folder layout (see README):

    datasets/<name>/train/*.png  and  datasets/<name>/val/*.png   (training data)
    datasets/<name>/*.png                                          (test data)
    datasets/<name>_scores/{train,val,test}.pt                     (made by generate_scores.py)

Scores are stored per image in the sorted order of `collect_images`, so the same
function must be used when generating and when loading them.
"""
from pathlib import Path

import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
MODES = ("train", "val", "test")


def collect_images(root):
    """Sorted list of all image files below `root` (hidden files such as .DS_Store are skipped)."""
    return sorted(
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS and not p.name.startswith(".")
    )


def split_root(dataset_path, mode):
    """Folder holding the images of `mode`: the dataset itself for test, a sub-folder otherwise."""
    dataset_path = Path(dataset_path)
    return dataset_path if mode == "test" else dataset_path / mode


def scores_path(dataset_path, mode):
    """Where the precomputed patch scores of `mode` live."""
    dataset_path = Path(dataset_path)
    return dataset_path.parent / f"{dataset_path.name}_scores" / f"{mode}.pt"


def build_transform(mode, input_size):
    """Resize to a square; train/val are ImageNet-normalised, test stays in [0, 1]."""
    steps = [
        transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ]
    if mode != "test":
        steps.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(steps)


class ImageScoreDataset(Dataset):
    def __init__(self, mode, dataset_path, input_size=224, patch_size=16):
        """
        Args:
            mode (str): "train", "val" or "test".
            dataset_path (str | Path): Dataset folder (see module docstring).
            input_size (int): Side length images are resized to.
            patch_size (int): Patch size of the model, used to validate the score files.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")

        self.transform = build_transform(mode, input_size)
        self.img_paths = collect_images(split_root(dataset_path, mode))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {split_root(dataset_path, mode)}")

        score_file = scores_path(dataset_path, mode)
        if not score_file.exists():
            raise RuntimeError(
                f"Score file '{score_file}' does not exist. Run generate_scores.py first.")
        self.scores = torch.load(score_file)

        expected = (len(self.img_paths), (input_size // patch_size) ** 2)
        if tuple(self.scores.shape) != expected:
            raise RuntimeError(
                f"Score file '{score_file}' has shape {tuple(self.scores.shape)} but "
                f"{expected} is expected for {len(self.img_paths)} images at input_size="
                f"{input_size}. Regenerate it with generate_scores.py --input_size {input_size}.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        orig_img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(orig_img), orig_img.size, self.scores[idx]
