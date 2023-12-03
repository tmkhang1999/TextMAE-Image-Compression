import argparse
import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from utils.distribution import cal_patch_score
from utils.map import Division_Merge_Segmented, laplacian


def preprocess_image_scores(dataset_path, output_file):
    dataset_path = Path(dataset_path)
    img_paths = sorted(dataset_path.rglob("*.*"))

    scores = list()
    for idx, img_path in tqdm(enumerate(img_paths)):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        s_map = Division_Merge_Segmented(img, (224, 224))
        t_map = laplacian(img, (224, 224))

        s_score = cal_patch_score(s_map)
        t_score = cal_patch_score(t_map)
        total_score = t_score * s_score

        if total_score.size > 0:
            total_score = (total_score - total_score.min()) / (total_score.max() - total_score.min())

        total_score = torch.tensor(total_score, dtype=torch.float32)
        scores.append(total_score)

    scores = torch.stack(scores)
    print("Shape of list_total_score: ", scores.shape)
    torch.save(scores, output_file)


def process_dataset(mode, dataset_path):
    dataset_path = Path(dataset_path)
    dir_path = dataset_path.parent
    dataset_name = dataset_path.name

    root = dataset_path if mode == "test" else dataset_path / mode

    output_folder = dir_path / f"{dataset_name}_scores"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f"{mode}.pt"

    preprocess_image_scores(root, output_file)


def main(args):
    training_path = Path(args.training_path)
    testing_path = Path(args.testing_path)

    process_dataset("train", training_path)
    process_dataset("val", training_path)
    process_dataset("test", testing_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and generate scores.")
    parser.add_argument("--training_path", required=True, help="Path to the training dataset.")
    parser.add_argument("--testing_path", required=True, help="Path to the testing dataset.")

    args = parser.parse_args()
    main(args)
