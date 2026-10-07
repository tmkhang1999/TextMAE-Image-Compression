"""
Patch selection: decide which patches of an image the encoder keeps.

Every strategy takes the per-patch importance scores (N, L) and returns
`ids_shuffle` (N, L): a permutation of the patch indices per image whose first
`num_keep` entries are the kept patches. The model derives `ids_restore` from it,
which is also what the decoder needs to put mask tokens back in place.
"""
import torch
import torch.nn.functional as F


def stratified_ids_shuffle(total_scores, num_keep):
    """
    Deterministic, score-stratified selection (the default).

    Scores are split into 10 buckets by percentile. The top bucket is always kept;
    the remaining quota is spread over the other buckets with a softmax over their
    mean score, keeping the highest-scoring patches inside each bucket.
    """
    num_patches = total_scores.shape[1]
    if num_keep > num_patches:
        raise ValueError(
            f"num_keep_patches ({num_keep}) is larger than the number of patches ({num_patches})"
        )

    percentiles = torch.arange(0.1, 0.91, 0.1, dtype=torch.float32, device=total_scores.device)
    top_bucket = len(percentiles)

    all_ids = []
    for score in total_scores:
        thresholds = torch.quantile(score.unique(), percentiles, dim=0)
        buckets = torch.bucketize(score, thresholds)

        bucket_means = torch.stack(
            [score[buckets == b].mean() for b in range(top_bucket + 1)]
        ).float().cpu()
        quota = torch.round(
            F.softmax(bucket_means[:-1], dim=0) * (num_keep - int((buckets == top_bucket).sum()))
        ).int()

        kept_values = score[buckets == top_bucket].tolist()
        for bucket, count in enumerate(quota):
            ranked, _ = torch.sort(score[buckets == bucket])
            kept_values.extend(ranked[len(ranked) - int(count):].tolist())

        # Turn the kept score values back into patch indices (ties are resolved in index order).
        ids, seen = [], set()
        remaining_freq = {}
        for value in kept_values:
            remaining_freq[value] = remaining_freq.get(value, 0) + 1
        for value, freq in remaining_freq.items():
            for idx in torch.nonzero(score == value).view(-1)[:freq].tolist():
                ids.append(idx)
                seen.add(idx)

        ids.extend(i for i in range(num_patches) if i not in seen)
        all_ids.append(ids)

    return torch.tensor(all_ids)


def multinomial_ids_shuffle(total_scores, num_keep, eps=1e-6):
    """
    Stochastic selection: sample patches without replacement, proportional to their score.

    `eps` keeps zero-score patches samplable (torch.multinomial needs at least as many
    non-zero weights as samples when sampling without replacement).
    """
    num_patches = total_scores.shape[1]
    if num_keep > num_patches:
        raise ValueError(
            f"num_keep_patches ({num_keep}) is larger than the number of patches ({num_patches})"
        )
    return torch.multinomial(total_scores + eps, num_samples=num_patches, replacement=False)


PATCH_SELECTORS = {
    "stratified": stratified_ids_shuffle,
    "multinomial": multinomial_ids_shuffle,
}


def select_patches(strategy, total_scores, num_keep):
    """Dispatch to a strategy in PATCH_SELECTORS and return `ids_shuffle` (N, L)."""
    if strategy not in PATCH_SELECTORS:
        raise ValueError(
            f"Unknown patch selection '{strategy}', choose from {sorted(PATCH_SELECTORS)}"
        )
    return PATCH_SELECTORS[strategy](total_scores, num_keep)
