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
    def __init__(self, is_train: bool, dataset_path: Path, transform):
        """
        Custom dataset for image data.

        Args:
            is_train (bool):
            dataset_path (Path): Path to the dataset.
            transform (torchvision.transforms.Compose): Image transformations.
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.root = os.path.join(self.dataset_path, "train" if is_train else "val")
        self.imgs_path = sorted(Path(self.root).rglob("*.*"))

        assert len(self.imgs_path) > 0, f"No images found in {dataset_path}"

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        orig_img = Image.open(img_path).convert("RGB")
        orig_shape = orig_img.size
        total_score = calculate_patch_score(orig_img)
        img = self.transform(orig_img)
        return img, orig_shape, total_score


def calculate_patch_score(img):
    img = np.array(img, dtype=np.uint8)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s_map = Division_Merge_Segmented(img, (224, 224))
    t_map = laplacian(img, (224, 224))

    s_score = cal_patch_score(s_map)
    t_score = cal_patch_score(t_map)

    total_score = t_score * s_score
    total_score = (total_score - total_score.min()) / (
        total_score.max() - total_score.min()
    )
    total_score = torch.tensor(total_score, dtype=torch.float32)

    return total_score


def get_image_dataset(is_train: bool, dataset_path: Path, args) -> Dataset:
    """
    Get an image dataset.

    Args:
        is_train (bool):
        dataset_path (str) : Dataset path.
        args (dict, optional): config.
    """
    if is_train:
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
    else:
        t = []
        t.append(
            transforms.Resize(224, interpolation=Image.BICUBIC)
        )  # to maintain same ratio w.r.t 224 images
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(t)

    dataset = CreateImageDataset(
        dataset_path=dataset_path, is_train=is_train, transform=transform
    )
    return dataset
