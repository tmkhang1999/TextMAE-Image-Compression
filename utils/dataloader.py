from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from utils.dataset_paths import known_datasets
from utils.mae_preprocessing import calculate_patch_score

__all__ = ["ImageDataset", "get_image_dataset"]  # Fix typo in the export name


class ImageDataset(Dataset):
    def __init__(self, dataset_path, transform):
        """
        Custom dataset for image data.

        Args:
            dataset_path (str): Path to the dataset.
            transform (torchvision.transforms.Compose): Image transformations.
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.imgs_path = sorted(Path(dataset_path).rglob("*.*"))
        assert len(self.imgs_path) > 0, f"No images found in {dataset_path}"

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        orig_img = Image.open(img_path)
        orig_shape = orig_img.size
        total_score = calculate_patch_score(orig_img)
        img = self.transform(orig_img)
        return img, orig_shape, total_score


def get_image_dataset(name: str, transform_cfg: dict = None) -> Dataset:
    """
    Get an image dataset.

    Args:
        name (str) : Dataset name.
        transform_cfg (dict, optional): Dictionary of transformation options.
    """
    transform = []

    if transform_cfg:
        if "crop" in transform_cfg:
            # Random crop with padding
            t = transforms.RandomCrop(
                transform_cfg["crop"], pad_if_needed=True, padding_mode="reflect"
            )
            transform.append(t)

        if transform_cfg.get("hflip", True):
            # Random horizontal flip
            t = transforms.RandomHorizontalFlip(p=0.5)
            transform.append(t)

    # Define a sequence of image transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        *transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(dataset_path=known_datasets.get(name, name), transform=transform)
    return dataset
