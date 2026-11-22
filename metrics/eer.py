"""Equal error rate (EER) for binary score-based detection."""

import numpy as np


def compute_det_curve(target_scores, nontarget_scores):
    """
    Detection error trade-off curve over every candidate threshold.

    A trial is accepted when its score is strictly above the threshold.
    Higher scores should therefore indicate the target class.

    Args:
        target_scores: 1-D array-like of scores for target (positive) trials.
        nontarget_scores: 1-D array-like of scores for non-target trials.

    Returns:
        frr: false rejection rate at each threshold.
        far: false acceptance rate at each threshold.
        thresholds: the thresholds, ascending; the first one lies just
            below the lowest score, so everything is accepted there.
    """
    target_scores = np.asarray(target_scores, dtype=float).ravel()
    nontarget_scores = np.asarray(nontarget_scores, dtype=float).ravel()
    n_target = target_scores.size
    n_nontarget = nontarget_scores.size

    scores = np.concatenate((target_scores, nontarget_scores))
    is_target = np.concatenate((np.ones(n_target), np.zeros(n_nontarget)))

    # Stable sort, so ties keep target trials ahead of non-target trials.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    is_target = is_target[order]

    # Raising the threshold past the k-th sorted score rejects the k
    # lowest-scoring trials.
    targets_rejected = np.cumsum(is_target)
    nontargets_rejected = np.arange(1, scores.size + 1) - targets_rejected
    nontargets_accepted = n_nontarget - nontargets_rejected

    frr = np.concatenate(([0.0], targets_rejected / n_target))
    far = np.concatenate(([1.0], nontargets_accepted / n_nontarget))
    thresholds = np.concatenate(([sorted_scores[0] - 0.001], sorted_scores))
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """
    Equal error rate and the threshold at which it occurs.

    The EER is taken where FRR and FAR are closest, as their mean.

    Returns:
        (eer, threshold)
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    best = np.argmin(np.abs(frr - far))
    return (frr[best] + far[best]) / 2, thresholds[best]
