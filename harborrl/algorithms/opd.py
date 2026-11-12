"""Fixed-prefix sampled-token reverse-KL surrogate, not full-vocabulary KL."""


def trajectory_surrogate(current, advantage, mask):
    import torch

    if current.shape != advantage.shape or current.shape != mask.shape or current.ndim != 2:
        raise ValueError("Expected aligned [trajectory, token] tensors")
    if not torch.isfinite(current).all() or not torch.isfinite(advantage).all():
        raise ValueError("Nonfinite loss input")
    if not ((mask == 0) | (mask == 1)).all() or (mask.sum(-1) == 0).any():
        raise ValueError("Each trajectory must have a nonempty binary mask")
    return -((current * advantage.detach() * mask).sum(-1) / mask.sum(-1)).mean()


def sampled_opd_loss(current, behavior, teacher, mask):
    if behavior.shape != current.shape or teacher.shape != current.shape:
        raise ValueError("OPD tensors must align")
    return trajectory_surrogate(current, teacher.detach() - behavior.detach(), mask)


def weighted_sampled_opd_loss(current, behavior, teacher, weights, mask):
    """Ragged backend sum; the shared scheduler supplies the outer group mean."""
    import torch

    if current.ndim != 1 or any(value.shape != current.shape for value in (behavior, teacher, weights, mask)):
        raise ValueError("OPD requires aligned flat token tensors")
    if any(not torch.isfinite(value).all() for value in (current, behavior, teacher, weights)):
        raise ValueError("Nonfinite OPD loss input")
    if (weights < 0).any() or not ((mask == 0) | (mask == 1)).all() or not mask.any():
        raise ValueError("OPD requires nonnegative weights and a nonempty binary mask")
    signal = teacher.detach() - behavior.detach()
    return -(current * signal * weights.detach() * mask.to(current.dtype)).sum()
