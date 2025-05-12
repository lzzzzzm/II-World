import math
import os
import numpy as np
from copy import deepcopy

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

from mmdet3d.core.hook.utils import is_parallel

__all__ = ['ScheduledSampling']


@HOOKS.register_module()
class ScheduledSampling(Hook):
    """Loss_Hook used in BEVDepth.
    """

    def __init__(self, total_iter=-1, k=2.5, loss_iter=None, trans_iter=None):
        super().__init__()
        self.total_iter = total_iter
        self.k = k
        self.loss_iter = loss_iter
        self.trans_iter = trans_iter

    def after_train_iter(self, runner):
        curr_step = runner.iter
        sample_rate = np.exp(- self.k * curr_step / self.total_iter)
        # sample_rate = 1 - curr_step / self.total_iter
        model = runner.model.module
        # update sample_rate
        model.sample_rate = sample_rate
        if self.loss_iter is not None:
            if curr_step == self.loss_iter:
                model.trajs_loss.loss_weight = model.trajs_loss.loss_weight * 10
        
        if self.trans_iter is not None:
            if curr_step == self.trans_iter:
                model.transformer.use_pred = True
                # print('Setting the transformation to use prediction')

