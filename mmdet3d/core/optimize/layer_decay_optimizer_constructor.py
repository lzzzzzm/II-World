# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

from mmcv.runner import DefaultOptimizerConstructor, get_dist_info

from mmdet.utils import get_root_logger
from ..builder import OPTIMIZER_BUILDERS


def get_layer_id_for_swin(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('encoders.camera.backbone.absolute_pos_embed'):
        return 0
    elif var_name.startswith('encoders.camera.backbone.patch_embed'):
        return 0
    elif var_name.startswith('encoders.camera.backbone.stages'):
        num_layers = [2, 2, 18, 2]
        stage_id = int(var_name.split('.')[4])
        if 'downsample' in var_name:
            return sum(num_layers[:stage_id + 1]) + 1
        block_id = int(var_name.split('.')[6])
        return sum(num_layers[:stage_id]) + block_id + 1
    else:
        return max_layer_id + 1


@OPTIMIZER_BUILDERS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = get_root_logger()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        logger.info('Build LearningRateDecayOptimizerConstructor  '
                    f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                if 'Swin' in module.forward_projection.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_swin(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            elif decay_type == 'stage_wise':
                raise NotImplementedError()
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructor(LearningRateDecayOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for BEiT,
    and it will be deprecated.
    Please use ``LearningRateDecayOptimizerConstructor`` instead.
    """

    def __init__(self, optimizer_cfg, paramwise_cfg):
        warnings.warn('DeprecationWarning: Original '
                      'LayerDecayOptimizerConstructor of BEiT '
                      'will be deprecated. Please use '
                      'LearningRateDecayOptimizerConstructor instead, '
                      'and set decay_type = layer_wise_vit in paramwise_cfg.')
        paramwise_cfg.update({'decay_type': 'layer_wise_vit'})
        warnings.warn('DeprecationWarning: Layer_decay_rate will '
                      'be deleted, please use decay_rate instead.')
        paramwise_cfg['decay_rate'] = paramwise_cfg.pop('layer_decay_rate')
        super(LayerDecayOptimizerConstructor,
              self).__init__(optimizer_cfg, paramwise_cfg)
