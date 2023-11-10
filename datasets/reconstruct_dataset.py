import os
import shutil
import argparse


def reorganize_folders(dataset_path):
    # Create destination directories
    directories = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if
                   os.path.isdir(os.path.join(dataset_path, d))]

    train_dest_path = os.path.join(dataset_path, 'train')
    val_dest_path = os.path.join(dataset_path, 'val')

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
            print(f"Moved: {src_file_path} to {dest_file_path}")

    for directory in directories:
        try:
            shutil.rmtree(directory, ignore_errors=True)
            print(f"Directory '{directory}' removed successfully.")
        except OSError as e:
            print(f"Error removing directory '{directory}': {e}")
    print("Reorganization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize folders in a dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")

    args = parser.parse_args()
    reorganize_folders(args.dataset_path)
