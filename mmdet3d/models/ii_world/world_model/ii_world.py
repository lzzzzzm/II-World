import random
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import DETECTORS, BACKBONES
from mmdet.models.utils import build_transformer

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder

import time

def compute_relative_rotation(ego_to_global_rotation):
    """
    将绝对旋转转换为相对旋转。
    :param ego_to_global_rotation: 形状为 [bs, f, 4] 的四元数张量
    :return: 形状为 [bs, f-1, 4] 的相对旋转张量
    """
    # 取出当前帧和前一帧的四元数
    current = ego_to_global_rotation[:, 1:, :]  # 从第2帧开始
    previous = ego_to_global_rotation[:, :-1, :]  # 到倒数第2帧结束

    # 计算前一帧的共轭（单位四元数的逆）
    previous_conjugate = previous.clone()
    previous_conjugate[:, :, 1:] *= -1  # 取负号 (x, y, z 分量)

    # 四元数乘法
    w1, x1, y1, z1 = previous_conjugate[..., 0], previous_conjugate[..., 1], previous_conjugate[..., 2], \
        previous_conjugate[..., 3]
    w2, x2, y2, z2 = current[..., 0], current[..., 1], current[..., 2], current[..., 3]

    relative_rotation = torch.zeros_like(current)
    relative_rotation[..., 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    relative_rotation[..., 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    relative_rotation[..., 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    relative_rotation[..., 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return relative_rotation


@DETECTORS.register_module()
class II_World(CenterPoint):
    def __init__(self,
                 # model module
                 vqvae=None,
                 transformer=None,
                 pose_encoder=None,
                 # params
                 previous_frame_exist=False,
                 previous_frame=None,
                 train_future_frame=None,
                 test_future_frame=None,
                 test_previous_frame=None,
                 observe_frame_number=4,
                 memory_frame_number=5,
                 sample_rate=1.0,
                 # Loss
                 feature_similarity_loss=None,
                 trajs_loss=None,
                 rotation_loss=None,
                 test_mode=False,
                 task_mode='generate',
                 **kwargs):
        super(II_World, self).__init__(**kwargs)
        # -------- Model Module --------
        self.pose_encoder = builder.build_head(pose_encoder)
        self.transformer = build_transformer(transformer)
        if test_mode:
            self.vqvae = builder.build_detector(vqvae)
        self.test_mode = test_mode

        # -------- Video Params --------
        self.observe_relative_rotation = None
        self.observe_delta_translation = None
        self.observe_ego_lcf_feat = None
        self.task_mode = task_mode

        # -------- Params --------
        self.previous_frame_exist = previous_frame_exist
        self.previous_frame = previous_frame if self.previous_frame_exist else 0
        self.train_future_frame = train_future_frame
        self.test_future_frame = test_future_frame
        self.test_previous_frame = test_previous_frame
        self.observe_frame_number = observe_frame_number + 1  # 2s+current frame default
        self.memory_frame_number = memory_frame_number
        self.sample_rate = sample_rate

        # -------- Loss -----------
        self.feature_similarity_loss = builder.build_loss(feature_similarity_loss)
        self.trajs_loss = builder.build_loss(trajs_loss)
        self.rotation_loss = builder.build_loss(rotation_loss)
        self.frame_loss_weight = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.05]  # as default
        self.occ_size = [200, 200, 16]
        self.foreground_cls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
        self.free_cls = 17
        self.save_index = 0

    def obtain_scene_from_token(self, token):
        if token.dim() == 4:
            token = token.unsqueeze(1)

        bs, f, c, w, h = token.shape
        decoder_shapes = [torch.tensor((200, 200)), torch.tensor((100, 100))]
        shapes = (bs, f, 200, 200, 16)
        token = token.view(bs * f, c, w, h)
        scene = self.vqvae.forward_decoder(token, decoder_shapes, shapes)
        return scene

    def load_transformation_info(self, img_metas, latent):
        device, dtype = latent.device, latent.dtype

        curr_to_future_ego_rt = torch.stack(
            [torch.tensor(img_meta['curr_to_future_ego_rt'], device=device, dtype=dtype) for img_meta in img_metas])
        curr_ego_to_global_rt = torch.stack(
            [torch.tensor(img_meta['curr_ego_to_global'], device=device, dtype=dtype) for img_meta in img_metas])
        ego_to_global_rotation = torch.stack(
            [torch.tensor(img_meta['ego_to_global_rotation'], device=device, dtype=dtype) for img_meta in img_metas])  # quarternion [bs, f, 4]
        ego_to_global_translation = torch.stack(
            [torch.tensor(img_meta['ego_to_global_translation'], device=device, dtype=dtype) for img_meta in img_metas])
        # Change translation to delta translation, only utilize x,y
        ego_to_global_delta_translation = ego_to_global_translation[:, 1:] - ego_to_global_translation[:, :-1]
        ego_to_global_delta_translation = ego_to_global_delta_translation[..., :2]
        # Compute relative rotation
        ego_to_global_relative_rotation = compute_relative_rotation(ego_to_global_rotation)

        gt_ego_lcf_feat = torch.stack(
            [torch.tensor(img_meta['gt_ego_lcf_feat'], device=device, dtype=dtype) for img_meta in img_metas])
        gt_ego_fut_cmd = torch.stack(
            [torch.tensor(img_meta['gt_ego_fut_cmd'], device=device, dtype=dtype) for img_meta in img_metas])
        start_of_sequence = torch.stack(
            [torch.tensor(img_meta['start_of_sequence'], device=device) for img_meta in img_metas])

        trans_infos = dict(
            curr_to_future_ego_rt=curr_to_future_ego_rt,
            curr_ego_to_global_rt=curr_ego_to_global_rt,
            ego_to_global_rotation=ego_to_global_rotation,
            ego_to_global_translation=ego_to_global_translation,
            ego_to_global_delta_translation=ego_to_global_delta_translation,
            ego_to_global_relative_rotation=ego_to_global_relative_rotation,
            gt_ego_lcf_feat=gt_ego_lcf_feat,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            # Sequence information
            start_of_sequence=start_of_sequence,
        )
        return trans_infos

    def process_observe_info(self, trans_infos, latent, start_update=True):
        bs, f, c, h, w = latent.shape
        start_of_sequence = trans_infos['start_of_sequence']
        device, dtype = latent.device, latent.dtype
        if start_update:
            if self.observe_relative_rotation is None:
                # Zero-init
                self.observe_relative_rotation =\
                    torch.ones(bs, self.observe_frame_number, 4, device=device,dtype=dtype)
                self.observe_delta_translation =\
                    torch.zeros(bs, self.observe_frame_number, 2, device=device,dtype=dtype)
                self.observe_ego_lcf_feat =\
                    torch.zeros(bs, self.observe_frame_number, 3, device=device, dtype=dtype)

            if start_of_sequence.sum() > 0:
                # Zero-init
                self.observe_relative_rotation[start_of_sequence] =\
                    torch.ones(start_of_sequence.sum(),self.observe_frame_number, 4, device=device, dtype=dtype)
                self.observe_delta_translation[start_of_sequence] =\
                    torch.zeros(start_of_sequence.sum(),self.observe_frame_number, 2, device=device, dtype=dtype)
                self.observe_ego_lcf_feat[start_of_sequence] =\
                    torch.zeros(start_of_sequence.sum(), self.observe_frame_number, 3, device=device, dtype=dtype)

        else:
            self.observe_delta_translation = torch.cat(
                [self.observe_delta_translation[:, 1:], trans_infos['ego_to_global_delta_translation'][:, 0:1]], dim=1)
            self.observe_relative_rotation = torch.cat(
                [self.observe_relative_rotation[:, 1:], trans_infos['ego_to_global_relative_rotation'][:, 0:1]], dim=1)
            self.observe_ego_lcf_feat = torch.cat(
                [self.observe_ego_lcf_feat[:, 1:], trans_infos['gt_ego_lcf_feat'][:, 0:1]], dim=1)

    def init_state(self, trans_infos, latent):
        bs, f, c, h, w = latent.shape
        device, dtype = latent.device, latent.dtype
        # As default memory_frame_number = 5, 4 history frames + 1 current frame
        history_token = latent[:, 0:1].repeat(1, self.memory_frame_number, 1, 1, 1).detach().clone()  # bs, f, c, w, h
        history_ego_lcf_feat = torch.zeros(bs, self.memory_frame_number, 3, device=device, dtype=dtype)
        history_relative_rotation = torch.ones(bs, self.memory_frame_number, 4, device=device, dtype=dtype)
        history_delta_translation = torch.zeros(bs, self.memory_frame_number, 2, device=device, dtype=dtype)

        history_relative_rotation[:, -self.observe_frame_number:] = self.observe_relative_rotation
        history_delta_translation[:, -self.observe_frame_number:] = self.observe_delta_translation
        history_ego_lcf_feat[:, -self.observe_frame_number:] = self.observe_ego_lcf_feat

        history_info = dict(
            history_token=history_token,
            history_ego_lcf_feat=history_ego_lcf_feat,
            history_relative_rotation=history_relative_rotation,
            history_delta_translation=history_delta_translation,
        )

        curr_latent = latent[:, 0].clone()
        curr_to_future_ego_rt = trans_infos['curr_to_future_ego_rt'][:, 0].clone()
        curr_ego_to_global = trans_infos['curr_ego_to_global_rt'][:, 0].clone()
        curr_rotation = trans_infos['ego_to_global_rotation'][:, 0].clone()
        curr_translation = trans_infos['ego_to_global_translation'][:, 0].clone()
        curr_ego_lcf_feat = trans_infos['gt_ego_lcf_feat'][:, 0].clone()
        curr_ego_mode = trans_infos['gt_ego_fut_cmd'][:, 0].clone()
        curr_relative_rotation = torch.ones(bs, 4, device=device, dtype=dtype)
        curr_delta_translation = torch.zeros(bs, 2, device=device, dtype=dtype)
        curr_info = dict(
            latent=latent,
            curr_latent=curr_latent,
            curr_to_future_ego_rt=curr_to_future_ego_rt,
            curr_ego_to_global=curr_ego_to_global,
            curr_rotation=curr_rotation,
            curr_translation=curr_translation,
            curr_ego_lcf_feat=curr_ego_lcf_feat,
            curr_ego_mode=curr_ego_mode,
            curr_relative_rotation=curr_relative_rotation,
            curr_delta_translation=curr_delta_translation
        )

        return history_info, curr_info

    def update_curr_info(self, curr_info, trans_infos, pred_trans_info, use_gt_rate, frame_idx, train):
        if self.task_mode == 'generate':
            curr_info['curr_to_future_ego_rt'] = trans_infos['curr_to_future_ego_rt'][:, frame_idx + 1]

        if train:
            curr_latent = torch.zeros_like(curr_info['curr_latent'])
            curr_latent[use_gt_rate] = curr_info['latent'][use_gt_rate, frame_idx + 1]
            curr_latent[~use_gt_rate] = pred_trans_info['pred_latent'][~use_gt_rate]
            curr_info['curr_latent'] = curr_latent

            curr_delta_translation = torch.zeros_like(curr_info['curr_delta_translation'])
            curr_delta_translation[use_gt_rate] = trans_infos['ego_to_global_delta_translation'][use_gt_rate, frame_idx]
            curr_delta_translation[~use_gt_rate] = pred_trans_info['pred_delta_translation'][~use_gt_rate]
            curr_info['curr_delta_translation'] = curr_delta_translation

            curr_translation = torch.zeros_like(curr_info['curr_translation'])
            curr_translation[use_gt_rate] = trans_infos['ego_to_global_translation'][use_gt_rate, frame_idx + 1]
            curr_translation[~use_gt_rate] = pred_trans_info['pred_next_translation'][~use_gt_rate]
            curr_info['curr_translation'] = curr_translation

            curr_rotation = torch.zeros_like(curr_info['curr_rotation'])
            curr_rotation[use_gt_rate] = trans_infos['ego_to_global_rotation'][use_gt_rate, frame_idx + 1]
            curr_rotation[~use_gt_rate] = pred_trans_info['pred_next_rotation'][~use_gt_rate]
            curr_info['curr_rotation'] = curr_rotation

            curr_relative_rotation = torch.zeros_like(curr_info['curr_relative_rotation'])
            curr_relative_rotation[use_gt_rate] = trans_infos['ego_to_global_relative_rotation'][use_gt_rate, frame_idx]
            curr_relative_rotation[~use_gt_rate] = pred_trans_info['pred_relative_rotation'][~use_gt_rate]
            curr_info['curr_relative_rotation'] = curr_relative_rotation

            curr_ego_to_global = torch.zeros_like(curr_info['curr_ego_to_global'])
            curr_ego_to_global[use_gt_rate] = trans_infos['curr_ego_to_global_rt'][use_gt_rate, frame_idx + 1]
            curr_ego_to_global[~use_gt_rate] = pred_trans_info['pred_next_ego_to_global'][~use_gt_rate]
            curr_info['curr_ego_to_global'] = curr_ego_to_global

            curr_ego_lcf_feat = torch.zeros_like(curr_info['curr_ego_lcf_feat'])
            curr_ego_lcf_feat[use_gt_rate] = trans_infos['gt_ego_lcf_feat'][use_gt_rate, frame_idx]
            curr_ego_lcf_feat[~use_gt_rate] = pred_trans_info['pred_ego_lcf_feat'][~use_gt_rate]
            curr_info['curr_ego_lcf_feat'] = curr_ego_lcf_feat

            curr_ego_mode = trans_infos['gt_ego_fut_cmd'][:, frame_idx]
            curr_info['curr_ego_mode'] = curr_ego_mode
        else:
            curr_info['curr_latent'] = pred_trans_info['pred_latent']
            curr_info['curr_delta_translation'] = pred_trans_info['pred_delta_translation']
            curr_info['curr_translation'] = pred_trans_info['pred_next_translation']
            curr_info['curr_relative_rotation'] = pred_trans_info['pred_relative_rotation']
            curr_info['curr_rotation'] = pred_trans_info['pred_next_rotation']
            curr_info['curr_ego_to_global'] = pred_trans_info['pred_next_ego_to_global']
            curr_info['curr_ego_lcf_feat'] = pred_trans_info['pred_ego_lcf_feat']
            curr_ego_mode = trans_infos['gt_ego_fut_cmd'][:, frame_idx]
            curr_info['curr_ego_mode'] = curr_ego_mode
        return curr_info

    def update_history_info(self, history_info, curr_info):
        history_info['history_token'] = torch.cat(
            [history_info['history_token'][:, 1:], curr_info['curr_latent'].unsqueeze(1).detach().clone()], dim=1)
        history_info['history_delta_translation'] = torch.cat(
            [history_info['history_delta_translation'][:, 1:], curr_info['curr_delta_translation'].unsqueeze(1).detach().clone()], dim=1)
        history_info['history_relative_rotation'] = torch.cat(
            [history_info['history_relative_rotation'][:, 1:], curr_info['curr_relative_rotation'].unsqueeze(1).detach().clone()], dim=1)
        history_info['history_ego_lcf_feat'] = torch.cat(
            [history_info['history_ego_lcf_feat'][:, 1:], curr_info['curr_ego_lcf_feat'].unsqueeze(1).detach().clone()], dim=1)
        return history_info


    def forward_sample(self, latent, img_metas, predict_future_frame, train=True, **kwargs):
        # latent: [bs, f, c, h, w]
        bs, f, c, h, w = latent.shape

        # -------------- Load GT Transformation --------------
        trans_infos = self.load_transformation_info(img_metas, latent)

        # ------------- History observe information -------------
        self.process_observe_info(trans_infos, latent, start_update=True)

        # -------------- Init hisotry & input tinformation --------------
        history_info, curr_info = self.init_state(trans_infos, latent)

        # ---------------- Autogressive prediction ----------------
        pred_latents = []
        pred_relative_rotations, pred_delta_translations = [], []

        for frame_idx in range(predict_future_frame):
            # Decide whether to use GT
            use_gt_rate = torch.rand(size=(bs,), device=latent.device) < self.sample_rate

            plan_query = self.pose_encoder.forward_encoder(history_info)

            pred_trans_info = self.transformer(
                curr_info=curr_info,
                history_info=history_info,
                plan_queries=plan_query,
            )

            pred_trans_info = self.pose_encoder.get_ego_feat(
                pred_trans_info=pred_trans_info,
                curr_info=curr_info,
                start_of_sequence=trans_infos['start_of_sequence']
            )

            if frame_idx != predict_future_frame - 1:
                # Update current info
                curr_info = self.update_curr_info(curr_info, trans_infos, pred_trans_info, use_gt_rate, frame_idx, train)
                # update history info
                history_info = self.update_history_info(history_info, curr_info)

            # Store the intermediate results
            pred_latents.append(pred_trans_info['pred_latent'])
            pred_delta_translations.append(pred_trans_info['pred_delta_translation'])
            pred_relative_rotations.append(pred_trans_info['pred_relative_rotation'])

        # Update observe information
        self.process_observe_info(trans_infos, latent, start_update=False)

        return_dict = dict(
            pred_latents=torch.stack(pred_latents, dim=1),                          # [bs, f, c, w, h], pred future latents
            pred_delta_translations=torch.stack(pred_delta_translations, dim=1),    # [bs, f, 2]
            pred_relative_rotations=torch.stack(pred_relative_rotations, dim=1),    # [bs, f, 4], pred future rotations
            targ_delta_translations=trans_infos['ego_to_global_delta_translation'], # [bs, f, 2], GT futuredelta translations
            targ_relative_rotations=trans_infos['ego_to_global_relative_rotation'], # [bs, f, 4], GT future rotations
        )
        return return_dict

    def forward_test(self, latent, voxel_semantics, img_metas, **kwargs):
        # Autoregressive predict future latent & Forward future latent
        sample_dict = self.forward_sample(latent, img_metas, self.test_future_frame, train=False)

        return_dict = dict()
        sample_idx = img_metas[0]['sample_idx']
        # Occupancy prediction
        if self.task_mode == 'generate':
            # Forward current latent
            pred_curr_voxel_semantics = self.obtain_scene_from_token(latent[:, 0])
            pred_curr_voxel_semantics = pred_curr_voxel_semantics.softmax(-1).argmax(-1)
            targ_future_voxel_semantics = voxel_semantics[:, self.test_previous_frame + 1:]
            targ_curr_voxel_semantics = voxel_semantics[:, self.test_previous_frame:self.test_previous_frame + 1]

            pred_latents = sample_dict['pred_latents']
            pred_voxel_semantics = self.obtain_scene_from_token(pred_latents)
            pred_voxel_semantics = pred_voxel_semantics.softmax(-1).argmax(-1)

            return_dict['pred_futu_semantics'] = pred_voxel_semantics.cpu().numpy().astype(np.uint8)
            return_dict['pred_curr_semantics'] = pred_curr_voxel_semantics.cpu().numpy().astype(np.uint8)
            return_dict['targ_futu_semantics'] = targ_future_voxel_semantics.cpu().numpy().astype(np.uint8)
            return_dict['targ_curr_semantics'] = targ_curr_voxel_semantics.cpu().numpy().astype(np.uint8)

        # Other information
        return_dict['occ_index'] = [img_meta['occ_index'] for img_meta in img_metas]
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        return_dict['sample_idx'] = sample_idx

        return [return_dict]

    def forward_train(self, latent, img_metas, **kwargs):
        # Forward auto-regressive prediction
        return_dict = self.forward_sample(latent, img_metas, self.train_future_frame, train=True)
        pred_latents = return_dict['pred_latents']
        targ_latents = latent[:, 1:]  # GT future latent

        pred_delta_translations = return_dict['pred_delta_translations']
        targ_delta_translations = return_dict['targ_delta_translations']

        pred_relative_rotations = return_dict['pred_relative_rotations']
        targ_relative_rotations = return_dict['targ_relative_rotations']

        # Get valid index for training
        valid_frame = torch.stack([torch.tensor(img_meta['valid_frame'], device=latent.device) for img_meta in img_metas])

        loss_dict = dict()
        for frame_idx in range(self.train_future_frame):
            loss_dict['feat_sim_{}s_loss'.format((frame_idx + 1) * 0.5)] = self.frame_loss_weight[frame_idx] * \
                                                                           self.feature_similarity_loss(
                                                                               pred_latents[:, frame_idx],
                                                                               targ_latents[:, frame_idx],
                                                                               valid_frame[:, frame_idx])

        loss_dict['trajs_loss'] = self.trajs_loss(pred_delta_translations, targ_delta_translations, valid_frame, None)
        loss_dict['rotation_loss'] = self.rotation_loss(pred_relative_rotations, targ_relative_rotations, valid_frame, None)

        return loss_dict
