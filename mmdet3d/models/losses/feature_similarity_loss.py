# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet3d.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

def MSE(pred, targ, valid, sample_weight=None):
    # pred: [bs, c, w, h]
    bs = pred.shape[0]
    if sample_weight is None:
        sample_weight = torch.ones(bs, device=pred.device, dtype=pred.dtype)

    valid_pred = pred * valid[:, None, None, None]
    valid_targ = targ * valid[:, None, None, None]
    loss = F.mse_loss(valid_pred, valid_targ)

    return loss

@LOSSES.register_module()
class FeatSimLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, targ, valid, sample_weight=None):
        # pred: [bs, c, w, h]
        bs = pred.shape[0]
        if sample_weight is None:
            sample_weight = torch.ones(bs, device=pred.device, dtype=pred.dtype)

        valid_pred = pred[valid]
        valid_targ = targ[valid]
        valid_sample_weight = sample_weight[valid]
        if valid_pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        mse_loss = torch.mean((valid_pred - valid_targ) ** 2, dim=(1, 2, 3))
        loss = self.loss_weight * (mse_loss * valid_sample_weight).mean()

        return loss
