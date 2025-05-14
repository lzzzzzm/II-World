import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

from mmdet.models.utils.builder import TRANSFORMER


def quaternion_to_rotation_matrix(quaternion):
    # Quterion to Rotation Matrix
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    batch_size = quaternion.size(0)
    R = torch.zeros((batch_size, 3, 3), device=quaternion.device, dtype=quaternion.dtype)

    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return R


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)


class MXUNet(nn.Module):
    def __init__(self, embed_dims, num_stages):
        super(MXUNet, self).__init__()
        self.embed_dims = embed_dims
        self.num_stages = num_stages

        self.recover_convs = nn.ModuleList()

        for i in range(num_stages):
            self.recover_convs.append(nn.Conv2d(embed_dims, embed_dims, 1))

    def forward(self, x, w, h):
        recover_x = torch.zeros_like(x[0])
        for i in range(self.num_stages):
            x_i = F.interpolate(x[i], size=(w, h), mode='bilinear', align_corners=False)
            recover_x += self.recover_convs[i](x_i)

        return recover_x


@TRANSFORMER.register_module()
class II_Former(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 low_encoder=None,
                 high_encoder=None,
                 positional_encoding=None,
                 embed_dims=256,
                 output_dims=256,
                 use_gt_traj=True,
                 use_transformation=True,
                 history_frame_number=5,
                 use_pred=False,
                 task_mode='generate',
                 **kwargs):
        super(II_Former, self).__init__(**kwargs)
        self.intra_encoder = build_transformer_layer_sequence(low_encoder)
        self.inter_decoder = build_transformer_layer_sequence(high_encoder)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.temporal_embedding = nn.Embedding(history_frame_number + 1, embed_dims)

        self.history_frame_number = history_frame_number
        self.embed_dims = embed_dims
        self.output_dims = output_dims
        self.use_gt_traj = use_gt_traj
        self.use_transformation = use_transformation
        self.use_pred = use_pred

        self.task_mode = task_mode

        # utilize U-Net to process multi-resolution features
        self.u_net = MXUNet(embed_dims, len(self.intra_encoder.layers))

        self.plan_embed = nn.Sequential(
            nn.Linear(12, embed_dims),
            nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims)
        )
        ego_mode = 3
        self.ego_mode = ego_mode
        self.plan_translation_output = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.Softplus(),
            nn.Linear(embed_dims // 2, (2+4) * ego_mode)
        )
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def process_transformation_matrix(self, bev_queries, curr_info, plan_output):
        bs = bev_queries.shape[0]

        curr_rotation = curr_info['curr_rotation']
        curr_translation = curr_info['curr_translation']
        curr_ego_to_global = curr_info['curr_ego_to_global']
        curr_to_future_ego_rt = curr_info['curr_to_future_ego_rt']

        pred_delta_translation = plan_output[:, :2]
        pred_relative_rotation = plan_output[:, 2:]
        pred_relative_rotation = F.normalize(pred_relative_rotation, p=2, dim=-1)
        pred_next_rotation = quaternion_multiply(pred_relative_rotation, curr_rotation)

        pred_next_translation = pred_delta_translation + curr_translation[:, :2]
        pred_next_translation = torch.cat([pred_next_translation, curr_translation[:, 2:]], dim=-1)

        pred_rotation_3x3 = quaternion_to_rotation_matrix(pred_next_rotation)
        pred_next_ego_to_global = torch.zeros(bs, 4, 4, device=bev_queries.device, dtype=bev_queries.dtype)
        pred_next_ego_to_global[:, :3, :3] = pred_rotation_3x3
        pred_next_ego_to_global[:, :3, 3] = pred_next_translation
        pred_next_ego_to_global[:, 3, 3] = 1

        pred_curr_to_futu = pred_next_ego_to_global.inverse() @ curr_ego_to_global
        pred_curr_to_futu_rotation = pred_curr_to_futu[:, :3, :3].reshape(bs, -1)
        pred_curr_to_futu_translation = pred_curr_to_futu[:, :3, 3].reshape(bs, -1)

        targ_curr_to_futu_rotation = curr_to_future_ego_rt[:, :3, :3].reshape(bs, -1)
        targ_curr_to_futu_translation = curr_to_future_ego_rt[:, :3, 3].reshape(bs, -1)

        targ_curr_to_futu_info = torch.cat([targ_curr_to_futu_rotation, targ_curr_to_futu_translation], dim=-1)
        pred_curr_to_futu_info = torch.cat([pred_curr_to_futu_rotation, pred_curr_to_futu_translation], dim=-1)

        trans_info = dict(
            pred_curr_to_futu_info=pred_curr_to_futu_info,
            targ_curr_to_futu_info=targ_curr_to_futu_info,
            # Predicted transformation matrix
            pred_delta_translation=pred_delta_translation,
            pred_next_translation=pred_next_translation,
            pred_relative_rotation=pred_relative_rotation,
            pred_next_rotation=pred_next_rotation,
            pred_next_ego_to_global=pred_next_ego_to_global,
        )
        return trans_info

    @force_fp32(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def forward(
            self,
            curr_info,
            history_info,
            plan_queries,
            **kwargs):
        """
        obtain bev features.
        input bev_queries: [bs, c, w, h]
        """
        bev_queries = curr_info['curr_latent']
        curr_ego_mode = curr_info['curr_ego_mode']
        hist_queries = history_info['history_token']

        bs, c, w, h = bev_queries.shape
        dtype = bev_queries.dtype

        # Process current queries
        bev_queries = bev_queries.reshape(bs, c, w * h).permute(0, 2, 1)  # change to [bs, w*h, c]
        bev_pos = self.positional_encoding(bs, w, h, bev_queries.device).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
        curr_temporal_embeddings = self.temporal_embedding.weight[-1][None].unsqueeze(-1)
        curr_temporal_embeddings = curr_temporal_embeddings.permute(0, 2, 1)

        # process history queries
        bs, f, c, w, h = hist_queries.shape
        history_temporal_embeddings = self.temporal_embedding.weight[0:self.history_frame_number][None].unsqueeze(-1).unsqueeze(-1)
        hist_queries = hist_queries + history_temporal_embeddings
        hist_queries = hist_queries.reshape(bs, c * f, w * h).permute(0, 2, 1)

        # add positional encoding
        bev_queries = bev_queries + bev_pos + curr_temporal_embeddings

        # intra_encoder process multi-resolution features
        bev_w, bev_h = [w], [h]
        for i in range(len(self.intra_encoder.layers)):
            if i == 0:
                bev_w.append(w)
                bev_h.append(h)
            else:
                bev_h.append((bev_h[-1] + 1) // 2)
                bev_w.append((bev_w[-1] + 1) // 2)

        bev_embed, plan_embed = self.intra_encoder(
            bev_queries,
            hist_queries,
            plan_queries=plan_queries,
            bev_w=bev_w,
            bev_h=bev_h,
            multi_scale=True,
            **kwargs
        )
        for i in range(len(bev_embed)):
            # recover to 2d
            w, h = bev_w[i + 1], bev_h[i + 1]
            bev_embed[i] = bev_embed[i].reshape(bs, w, h, -1).permute(0, 3, 1, 2)

        bev_embed = self.u_net(bev_embed, bev_w[0], bev_h[0])
        bev_embed = bev_embed.reshape(bs, c, -1).permute(0, 2, 1)

        # Pred transformation matrix
        if self.ego_mode == 1:
            plan_output = self.plan_translation_output(plan_embed[-1].squeeze(1))
        else:
            plan_output = self.plan_translation_output(plan_embed[-1].squeeze(1)).reshape(bs, -1, 6)
            plan_output = plan_output[curr_ego_mode.bool()]

        trans_info = self.process_transformation_matrix(bev_queries, curr_info, plan_output)
        if self.task_mode == 'generate':
            curr_to_futu_info = trans_info['targ_curr_to_futu_info']

        bev_embed = bev_embed + self.plan_embed(curr_to_futu_info).unsqueeze(1)

        w, h = bev_w[0], bev_h[0]
        bev_w, bev_h = [], []
        for i in range(len(self.inter_decoder.layers)):
            bev_w.append(w)
            bev_h.append(h)
        bev_embed, _ = self.inter_decoder(
            bev_embed,
            hist_queries,
            plan_queries=None,
            bev_w=bev_w,
            bev_h=bev_h,
            multi_scale=False,
            **kwargs
        )

        bev_embed = bev_embed.permute(0, 2, 1).reshape(bs, -1, w, h)

        trans_info['pred_latent'] = bev_embed
        return trans_info

