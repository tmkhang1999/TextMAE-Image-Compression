import os
import shutil
import argparse


def reorganize_folders(dataset_path, dest_path):
    os.makedirs(dest_path, exist_ok=True)

    train_dest_path = os.path.join(dest_path, 'train')
    val_dest_path = os.path.join(dest_path, 'val')

    os.makedirs(train_dest_path, exist_ok=True)
    os.makedirs(val_dest_path, exist_ok=True)

    # Walk through the source directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Ignore files in the root directory
            if root == dataset_path:
                continue

            # Determine the destination directory
            if 'train' in root:
                dest_dir = train_dest_path
            elif 'val' in root:
                dest_dir = val_dest_path

            # Copy the file to the destination directory
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, file)
            shutil.copy(src_file_path, dest_file_path)
            shutil.rmtree(src_file_path, ignore_errors=True)
            print(f"Moved: {src_file_path} to {dest_file_path}")

    try:
        shutil.rmtree(dataset_path, ignore_errors=True)
        print(f"Directory '{dataset_path}' removed successfully.")
    except OSError as e:
        print(f"Error removing directory '{dataset_path}': {e}")

    print("Reorganization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize folders in a dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("dest_path", type=str, help="Path to organize the dataset")

    args = parser.parse_args()
    reorganize_folders(args.dataset_path, args.dest_path)
