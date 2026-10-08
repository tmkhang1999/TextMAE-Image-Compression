"""
Patch selection: decide which patches of an image the encoder keeps.

Every strategy takes the per-patch importance scores (N, L) and returns
`ids_shuffle` (N, L): a permutation of the patch indices per image whose first
`num_keep` entries are the kept patches. The model derives `ids_restore` from it,
which is also what the decoder needs to put mask tokens back in place.
"""
import torch
import torch.nn.functional as F


def _rank_desc(indices, score):
    """Order `indices` by descending score; equal scores are ordered at random."""
    shuffled = indices[torch.randperm(len(indices), device=indices.device)]
    order = torch.sort(score[shuffled], descending=True, stable=True).indices
    return shuffled[order]


def stratified_ids_shuffle(total_scores, num_keep):
    """
    Score-stratified ("percentile") selection, the default.

    Scores are split into 10 percentile buckets. The top bucket is always kept; the
    remaining budget is shared by the other buckets in proportion to the softmax of
    their mean score, taking the best patches of each bucket. Any budget left after
    rounding goes to the best remaining patches. Ties are broken at random, so large
    areas of equal (e.g. zero) score are not filled from the top of the image.
    """
    num_patches = total_scores.shape[1]
    if num_keep > num_patches:
        raise ValueError(
            f"num_keep_patches ({num_keep}) is larger than the number of patches ({num_patches})"
        )

    device = total_scores.device
    percentiles = torch.arange(0.1, 0.91, 0.1, dtype=torch.float32, device=device)
    top_bucket = len(percentiles)

    all_ids = []
    for score in total_scores:
        thresholds = torch.quantile(score.unique().float(), percentiles)
        buckets = torch.bucketize(score, thresholds)

        top = torch.nonzero(buckets == top_bucket).view(-1)
        budget = max(num_keep - len(top), 0)

        # Empty buckets get -inf so they receive no share of the budget
        means = torch.stack([
            score[buckets == b].mean() if (buckets == b).any() else torch.tensor(float("-inf"), device=device)
            for b in range(top_bucket)
        ]).float()
        quota = torch.round(F.softmax(means, dim=0) * budget).long().tolist()

        chosen = [top]
        for bucket, count in enumerate(quota):
            members = torch.nonzero(buckets == bucket).view(-1)
            chosen.append(_rank_desc(members, score)[:count])
        chosen = _rank_desc(torch.cat(chosen), score)[:num_keep]

        is_chosen = torch.zeros(num_patches, dtype=torch.bool, device=device)
        is_chosen[chosen] = True
        rest = _rank_desc(torch.nonzero(~is_chosen).view(-1), score)
        all_ids.append(torch.cat([chosen, rest]))

    return torch.stack(all_ids).cpu()


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
