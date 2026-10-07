"""
Flatten a Kaggle ImageNet-100 download into datasets/<name>/{train,val}/.

Usage: python tools/prepare_imagenet.py <downloaded_folder> <output_folder> [--remove-source]
"""
import argparse
import shutil
from pathlib import Path


def reorganize_folders(dataset_path, dest_path, remove_source=False):
    dataset_path, dest_path = Path(dataset_path), Path(dest_path)
    destinations = {"train": dest_path / "train", "val": dest_path / "val"}
    for folder in destinations.values():
        folder.mkdir(parents=True, exist_ok=True)

    for src in sorted(dataset_path.rglob("*")):
        # Skip folders and files sitting directly in the root
        if not src.is_file() or src.parent == dataset_path:
            continue

        split = next((name for name in destinations if name in src.parent.parts), None)
        if split is None:
            print(f"Skipped (not under a train/val folder): {src}")
            continue

        shutil.copy(src, destinations[split] / src.name)

    if remove_source:
        shutil.rmtree(dataset_path)
        print(f"Removed '{dataset_path}'")
    print("Reorganization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("dataset_path", help="Downloaded dataset folder")
    parser.add_argument("dest_path", help="Output folder that gets train/ and val/")
    parser.add_argument("--remove-source", action="store_true",
                        help="Delete dataset_path after copying")
    args = parser.parse_args()
    reorganize_folders(args.dataset_path, args.dest_path, args.remove_source)
