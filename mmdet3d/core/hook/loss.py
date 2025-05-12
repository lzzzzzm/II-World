# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
from copy import deepcopy

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

from mmdet3d.core.hook.utils import is_parallel

__all__ = ['Loss_Hook']


@HOOKS.register_module()
class Loss_Hook(Hook):
    """Loss_Hook used in BEVDepth.
    """

    def __init__(self, total_iter=-1):
        super().__init__()
        self.total_iter = total_iter

    def after_train_iter(self, runner):
        runner.ema_model.update(runner, runner.model.module)
        curr_step = runner.iter
        loss_weight = max(0.2, (1.0 - curr_step / self.total_iter))
        model = runner.model.module
        model.sem_scal_loss_weight = loss_weight
