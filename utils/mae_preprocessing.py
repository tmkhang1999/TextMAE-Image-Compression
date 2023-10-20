import numpy as np
from collections import Counter
import math

from utils.distribution import cal_patch_score
from utils.map import Division_Merge_Segmented, laplacian


def get_filtered_indices(scores, keep_ratio=0.2):
    sorted_scores = np.sort(scores)

    # Calculate percentiles and thresholds
    percentiles = np.arange(10, 91, 10)
    thresholds = np.percentile(np.unique(sorted_scores), percentiles)

    # Categorize data into groups
    categories = np.digitize(sorted_scores, thresholds)

    # Calculate group means
    group_means = [np.mean(sorted_scores[categories == group]) for group in range(len(percentiles) + 1)]

    # Keep values from the group with highest category (categorized_data == 9)
    keep_values = list(sorted_scores[categories == 9])

    # Apply softmax to group means for other groups
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    softmaxed_means = softmax(group_means[:-1])  # Exclude the last group
    new_target = math.ceil(keep_ratio * len(sorted_scores) - len(keep_values))
    scaled_means = np.round(softmaxed_means * new_target)

    # Populate high_category_values
    for i, num_to_keep in enumerate(scaled_means):
        start_index = len(sorted_scores[categories == i]) - num_to_keep
        keep_values.extend(list(sorted_scores[categories == i][int(start_index):]))

    keep_values.append(sorted_scores[0])  # Append the least important patch
    keep_values_frequency = Counter(keep_values)
    indices = []

    # Create a list of indices
    for value, freq in keep_values_frequency.items():
        indices.extend(list(np.where(scores == value)[0][:freq]))

    remaining_indices = [i for i in range(len(scores)) if i not in indices]
    indices.extend(remaining_indices)

    return indices


def calculate_patch_score(img):
    s_map = Division_Merge_Segmented(img, (224, 224))
    t_map = laplacian(img, (224, 224))

    s_score = cal_patch_score(s_map)
    t_score = cal_patch_score(t_map)

    total_score = t_score * s_score

    return total_score
