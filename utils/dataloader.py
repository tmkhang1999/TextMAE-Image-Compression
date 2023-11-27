import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.distribution import cal_patch_score
from utils.map import Division_Merge_Segmented, laplacian

import cv2
import numpy as np

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CreateImageDataset(Dataset):
    def __init__(self, mode: str, dataset_path: Path, scores_file: Path, transform):
        """
        Custom dataset for image data.

        Args:
            mode (str): Dataset mode ("train", "val", or "test").
            dataset_path (Path): Path to the dataset.
            transform (torchvision.transforms.Compose): Image transformations.
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.root = self.dataset_path if mode == "test" else os.path.join(self.dataset_path, mode)
        self.imgs_path = sorted(Path(self.root).rglob("*.*"))

        assert len(self.imgs_path) > 0, f"No images found in {dataset_path}"

        with open(scores_file, 'r') as f:
            self.scores = json.load(f)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        orig_img = Image.open(img_path).convert("RGB")
        orig_shape = orig_img.size
        total_score = self.scores[str(idx)]
        img = self.transform(orig_img)
        return img, orig_shape, torch.tensor(total_score, dtype=torch.float32)


def get_image_dataset(mode: str, dataset_path: Path, scores_file: Path, args) -> Dataset:
    """
    Get an image dataset.

    Args:
        mode (str): Dataset mode ("train", "val", or "test").
        dataset_path (Path): Dataset path.
        scores_file (Path): The file saving the total score of images in dataset.
        args (dict, optional): config.
    """
    assert mode in ["train", "val", "test"], "Mode must be one of ['train', 'val', 'test']"

    if mode == "train":
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    elif mode == "val":
        t = list()
        t.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))  # to maintain same ratio w.r.t 224 images
        # t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(t)
    else:
        t = list()
        t.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)

    dataset = CreateImageDataset(
        dataset_path=dataset_path, scores_file=scores_file, mode=mode, transform=transform
    )
    return dataset
