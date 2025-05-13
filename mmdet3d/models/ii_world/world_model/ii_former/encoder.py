import numpy as np

import torch
import torch.nn as nn
import copy
import warnings

from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import ext_loader

from .custom_layer_sequence import CustomTransformerLayerSequence
from .transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class II_FormerEncoder(CustomTransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 return_intermediate=False,
                 *args,
                 **kwargs):

        super(II_FormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32()
    def forward(self,
                bev_query,
                hist_queries,
                plan_queries=None,
                bev_h=None,
                bev_w=None,
                multi_scale=False,
                **kwargs):

        ref_2d_list = []
        for w, h in zip(bev_w, bev_h):
            ref_2d = self.get_reference_points(h, w, dim='2d', bs=bev_query.size(0), device=bev_query.device, dtype=bev_query.dtype)
            ref_2d_list.append(ref_2d)

        output = bev_query
        output_plan = plan_queries
        intermediate, intermediate_plan = [], []
        for lid, layer in enumerate(self.layers):

            output, output_plan = layer(
                bev_query,
                hist_queries,
                plan_queries=plan_queries,
                bev_w=bev_w[lid],
                bev_h=bev_h[lid],
                ref_2d=ref_2d_list[lid],
                downsample=False if lid==0 else True,   # only work with conv_cfgs
                **kwargs
            )

            bev_query = output
            plan_queries = output_plan

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_plan.append(output_plan)

        if self.return_intermediate:
            return intermediate, intermediate_plan

        return output, output_plan

@TRANSFORMER_LAYER.register_module()
class II_FormerEncoderLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels=512,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 memory_w=8,
                 memory_h=8,
                 **kwargs):
        super(II_FormerEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        self.memory_w = memory_w
        self.memory_h = memory_h
        # assert len(operation_order) in {2, 4, 6}

    @force_fp32()
    def forward(self,
                query,
                hist_queries,
                plan_queries=None,
                bev_h=None,
                bev_w=None,
                ref_2d=None,
                prev_bev=None,
                level_start_index=None,
                downsample=False,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        conv_index = 0
        bs, num_query, dims = query.shape
        identity = query
        plan_identity = plan_queries

        for layer in self.operation_order:
            # self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    None,
                    None,
                    identity if self.pre_norm else None,
                    reference_points=ref_2d,
                    level_start_index=torch.tensor([0], device=query.device),
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                plan_queries = self.plan_norms[norm_index](plan_queries) if plan_queries is not None else None
                norm_index += 1

            # cross attention
            elif layer == 'cross_attn':
                query, plan_queries = self.attentions[attn_index](
                    query,
                    hist_queries=hist_queries,
                    plan_queries=plan_queries,
                    **kwargs)
                attn_index += 1
                identity = query
                plan_identity = plan_queries

            elif layer == 'temporal_fusion':
                query, plan_queries = self.attentions[attn_index](
                    query,
                    hist_queries=hist_queries,
                    plan_queries=plan_queries,
                    **kwargs)
                attn_index += 1
                identity = query
                plan_identity = plan_queries

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                plan_queries = self.plan_ffns[ffn_index](plan_queries, plan_identity if self.pre_norm else None) if plan_queries is not None else None
                ffn_index += 1

            elif layer == 'conv':
                query = query.reshape(bs, bev_h, bev_w, dims).permute(0, 3, 1, 2)
                query = self.convs[conv_index](query)
                query = query.permute(0, 2, 3, 1).reshape(bs, -1, dims)
                conv_index += 1
            else:
                raise ValueError(f'Invalid operation order {layer}')

        return query, plan_queries
