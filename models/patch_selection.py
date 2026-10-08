"""Patch selection: which patches the encoder keeps.

Each strategy maps per-patch scores (N, L) to `ids_shuffle` (N, L), a permutation of the patch
indices per image whose first `num_keep` entries are the kept patches.
"""
import torch
import torch.nn.functional as F


def _rank_desc(indices, score):
    """Indices ordered by descending score; equal scores are ordered at random."""
    shuffled = indices[torch.randperm(len(indices), device=indices.device)]
    return shuffled[torch.sort(score[shuffled], descending=True, stable=True).indices]


def _check(total_scores, num_keep):
    if num_keep > total_scores.shape[1]:
        raise ValueError(f"num_keep_patches ({num_keep}) is larger than the number of patches ({total_scores.shape[1]})")


def stratified_ids_shuffle(total_scores, num_keep):
    """Percentile sampling (default): always keep the top score bucket, share the rest of the budget across
    the other buckets by the softmax of their mean score and take the best patches of each. Ties are broken
    at random so equal (e.g. zero) scores are not filled from the top of the image."""
    _check(total_scores, num_keep)
    device, num_patches = total_scores.device, total_scores.shape[1]
    percentiles = torch.arange(0.1, 0.91, 0.1, dtype=torch.float32, device=device)
    top_bucket = len(percentiles)

    all_ids = []
    for score in total_scores:
        buckets = torch.bucketize(score, torch.quantile(score.unique().float(), percentiles))
        top = torch.nonzero(buckets == top_bucket).view(-1)
        # empty buckets get -inf, i.e. no share of the budget
        means = torch.stack([score[buckets == b].mean() if (buckets == b).any() else torch.tensor(float("-inf"), device=device)
                             for b in range(top_bucket)]).float()
        quota = torch.round(F.softmax(means, dim=0) * max(num_keep - len(top), 0)).long().tolist()

        chosen = [top] + [_rank_desc(torch.nonzero(buckets == b).view(-1), score)[:q] for b, q in enumerate(quota)]
        chosen = _rank_desc(torch.cat(chosen), score)[:num_keep]
        is_chosen = torch.zeros(num_patches, dtype=torch.bool, device=device)
        is_chosen[chosen] = True
        all_ids.append(torch.cat([chosen, _rank_desc(torch.nonzero(~is_chosen).view(-1), score)]))
    return torch.stack(all_ids).cpu()


def multinomial_ids_shuffle(total_scores, num_keep, eps=1e-6):
    """Sample patches without replacement, proportional to their score (eps keeps zero scores samplable)."""
    _check(total_scores, num_keep)
    return torch.multinomial(total_scores + eps, num_samples=total_scores.shape[1], replacement=False)


PATCH_SELECTORS = {"stratified": stratified_ids_shuffle, "multinomial": multinomial_ids_shuffle}


def select_patches(strategy, total_scores, num_keep):
    if strategy not in PATCH_SELECTORS:
        raise ValueError(f"Unknown patch selection '{strategy}', choose from {sorted(PATCH_SELECTORS)}")
    return PATCH_SELECTORS[strategy](total_scores, num_keep)
