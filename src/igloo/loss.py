import torch
import torch.nn as nn


def mse_loss(pred, target):
    """Mean Squared Error"""
    return nn.functional.mse_loss(pred, target)

def mae_loss(pred, target):
    """Mean Absolute Error"""
    return nn.functional.l1_loss(pred, target)

def huber_loss(pred, target, delta=1.0):
    """Huber loss (smooth L1)"""
    return nn.functional.smooth_l1_loss(pred, target, beta=delta)

def quantile_loss(pred, target, quantile=0.5):
    """Quantile (pinball) loss"""
    diff = target - pred
    return torch.max(quantile * diff, (quantile - 1) * diff).mean()

def weighted_loss(pred, target, loss_fn, weights=None, **kwargs):
    """Apply weights to any loss function (e.g., for sample weighting)."""
    loss = loss_fn(pred, target, **kwargs)
    if weights is not None:
        loss = (loss.squeeze() * weights.squeeze()).mean()
    return loss

def combined_loss(pred, target, loss_fns, weights=None, **kwargs):
    """Combine multiple loss functions with weights.
	loss_fns: list of (fn, kwarg_dict)
	weights: list of floats
	"""
    total = 0.0
    for i, (fn, kw) in enumerate(loss_fns):
        w = weights[i] if weights is not None else 1.0
        total += w * fn(pred, target, **kw)
    return total
