import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_prob, target, eps=1e-6):
    pred_flat = pred_prob.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def segmentation_loss(logits, target):
    logits = logits.squeeze(1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(torch.sigmoid(logits), target)