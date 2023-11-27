import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from utils.dataloader import cal_patch_score
from utils.map import Division_Merge_Segmented, laplacian


def preprocess_image_scores(dataset_path, output_file):
    dataset_path = Path(dataset_path)
    img_paths = sorted(dataset_path.rglob("*.*"))

    scores = {}
    for idx, img_path in tqdm(enumerate(img_paths)):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        s_map = Division_Merge_Segmented(img, (224, 224))
        t_map = laplacian(img, (224, 224))

        s_score = cal_patch_score(s_map)
        t_score = cal_patch_score(t_map)
        total_score = t_score * s_score

        if total_score.size > 0:
            total_score = (total_score - total_score.min()) / (total_score.max() - total_score.min())
        scores[idx] = total_score.tolist()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(scores, f)


if __name__ == "__main__":
    preprocess_image_scores("./kodak", "./kodak_score.json")
