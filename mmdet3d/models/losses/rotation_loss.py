# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet3d.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

def quaternion_inverse(quaternion):
    """Compute the inverse of a quaternion."""
    q_conj = quaternion * torch.tensor([1, -1, -1, -1], dtype=quaternion.dtype, device=quaternion.device)
    q_norm = torch.sum(quaternion ** 2, dim=-1, keepdim=True)
    return q_conj / q_norm

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def compute_relative_rotations(quaternions):
    """Compute relative rotations from a sequence of quaternions."""
    relative_rotations = []
    for i in range(1, len(quaternions)):
        q_inv = quaternion_inverse(quaternions[i-1])
        relative_rotation = quaternion_multiply(q_inv, quaternions[i])
        relative_rotations.append(relative_rotation)
    return torch.stack(relative_rotations)

def compute_cumulative_rotations(relative_rotations):
    """Compute cumulative rotations from relative rotations."""
    cumulative_rotations = [relative_rotations[0]]
    for i in range(1, len(relative_rotations)):
        cumulative_rotation = quaternion_multiply(cumulative_rotations[-1], relative_rotations[i])
        cumulative_rotations.append(cumulative_rotation)
    return torch.stack(cumulative_rotations)


@LOSSES.register_module()
class RotationLoss(nn.Module):
    def __init__(self, loss_weight=5.0, translation_loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.translation_loss_weight = translation_loss_weight
        self.frame_loss_weight = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]  # as default
        self.frame_loss_weight = torch.tensor(self.frame_loss_weight, dtype=torch.float32, device='cuda')

    def quaternion_similarity_loss(self, q1, q2):
        q1 = q1 / q1.norm(dim=1, keepdim=True)
        q2 = q2 / q2.norm(dim=1, keepdim=True)

        # 计算四元数的点积
        dot_product = torch.sum(q1 * q2, dim=1).abs()

        # 计算损失
        loss = 1 - dot_product
        return loss

    def forward(self, pred_rotations, targ_rotations, valid, sample_weight=None):
        # simple_pred_rotations = targ_rotations[:, 0:1].repeat(1, pred_rotations.size(1), 1)

        # cumulative rotation loss
        bs = pred_rotations.shape[0]
        quaternion_similarity_loss = 0
        for i in range(bs):
            quaternion_similarity_loss += self.quaternion_similarity_loss(pred_rotations[i], targ_rotations[i])


        dot_product = torch.sum(pred_rotations * targ_rotations, dim=-1)
        dot_product = torch.clamp(dot_product, -1, 1)

        loss = 1.0 - torch.abs(dot_product)
        return loss.sum(-1).mean() * self.loss_weight