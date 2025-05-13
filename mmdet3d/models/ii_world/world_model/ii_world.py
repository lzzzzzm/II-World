# Modified from OccWorld
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
                 **kwargs):
        super(II_World, self).__init__(**kwargs)
        # -------- Model Module --------
        if not self.training:
            self.vqvae = builder.build_detector(vqvae)
        self.pose_encoder = builder.build_head(pose_encoder)
        self.transformer = build_transformer(transformer)

        # -------- Video Params --------
        self.observe_rotation = None
        self.observe_relative_rotation = None
        self.observe_delta_translation = None
        self.observe_ego_lcf_feat = None
        self.observe_ego_mode = None
        self.observe_curr_to_futu = None
        self.observe_plan_embed = None

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

    def decode_pose(self, pose_mat, ego_to_globals, ego_to_lidar=None):
        bs, f = pose_mat.shape[:2]
        # check type to numpy
        if isinstance(ego_to_globals, torch.Tensor):
            ego_to_globals = ego_to_globals.cpu().numpy()
        if isinstance(pose_mat, torch.Tensor):
            pose_mat = pose_mat.cpu().numpy()
        if ego_to_lidar is not None:
            if isinstance(ego_to_lidar, torch.Tensor):
                ego_to_lidar = ego_to_lidar.cpu().numpy()
        global_to_ego = np.linalg.inv(ego_to_globals)

        outs = []
        for i in range(bs):
            trajs = []
            for j in range(f - 1):
                curr_xyz = pose_mat[i, j, :2, 3]
                next_xyz = pose_mat[i, j + 1, :2, 3]
                global_pose = np.array([curr_xyz, next_xyz])
                global_pose = np.concatenate([global_pose, np.zeros((global_pose.shape[0], 1))], axis=1)
                # trans to ego
                global_pose = np.concatenate([global_pose, np.ones((global_pose.shape[0], 1))], axis=1)
                global_pose = np.dot(global_to_ego[i, j], global_pose.T).T

                # trans to lidar
                if ego_to_lidar is not None:
                    global_pose = np.dot(ego_to_lidar[i, j], global_pose.T).T

                # get trajs
                ego_trajs = global_pose[1:] - global_pose[:-1]
                trajs.append(ego_trajs[:, :2].squeeze())
            outs.append(trajs)
        return np.array(outs)

    def forward_sample(self, latent, img_metas, predict_future_frame, use_gt, train=True, **kwargs):
        # latent: [bs, f, c, h, w]
        bs, f, c, h, w = latent.shape

        # -------------- Load GT Transformation --------------
        # Training information
        curr_to_future_ego_rt = torch.stack(
            [torch.tensor(img_meta['curr_to_future_ego_rt'], device=latent.device, dtype=torch.float32) for img_meta in
             img_metas]
        )
        curr_ego_to_global_rt = torch.stack(
            [torch.tensor(img_meta['curr_ego_to_global'], device=latent.device, dtype=torch.float32) for img_meta in
             img_metas]
        )
        ego_to_global_rotation = torch.stack(
            [torch.tensor(img_meta['ego_to_global_rotation'], device=latent.device, dtype=torch.float32) for img_meta in
             img_metas]
        )  # quarternion [bs, f, 4]
        ego_to_global_translation = torch.stack(
            [torch.tensor(img_meta['ego_to_global_translation'], device=latent.device, dtype=torch.float32) for img_meta
             in img_metas]
        )
        # Change translation to delta translation, only utilize x,y
        ego_to_global_delta_translation = ego_to_global_translation[:, 1:] - ego_to_global_translation[:, :-1]
        ego_to_global_delta_translation = ego_to_global_delta_translation[..., :2]
        # Compute relative rotation
        ego_to_global_relative_rotation = compute_relative_rotation(ego_to_global_rotation)

        gt_ego_lcf_feat = torch.stack(
            [torch.tensor(img_meta['gt_ego_lcf_feat'], device=latent.device, dtype=torch.float32) for img_meta in
             img_metas]
        )
        gt_ego_fut_cmd = torch.stack(
            [torch.tensor(img_meta['gt_ego_fut_cmd'], device=latent.device, dtype=torch.float32) for img_meta in
             img_metas]
        )
        # only for inference
        if not train:
            ego_from_sensor = torch.stack(
                [torch.tensor(img_meta['ego_from_sensor'], device=latent.device, dtype=torch.float32) for img_meta in
                 img_metas]
            )
            targ_pose_mat = torch.stack(
                [torch.tensor(img_meta['pose_mat'], device=latent.device, dtype=torch.float32) for img_meta in
                 img_metas]
            )
            # when inference, only support batch size = 1 now
            assert bs == 1
            gt_bboxes_3d = img_metas[0]['gt_bboxes_3d']
            gt_attr_labels = img_metas[0]['gt_attr_labels']
            ego_to_lidar = torch.stack(
                [torch.tensor(img_meta['ego_to_lidar'], device=latent.device, dtype=torch.float32) for img_meta in
                 img_metas]
            )

        # ------------- History observe information -------------
        start_of_sequence = torch.stack(
            [torch.tensor(img_meta['start_of_sequence'], device=latent.device) for img_meta in img_metas])
        # Deal with first frame
        if self.observe_rotation is None:
            self.observe_rotation = ego_to_global_rotation[:, 0:1].repeat(1, self.observe_frame_number, 1)
            self.observe_relative_rotation = torch.ones(bs, self.observe_frame_number, 4, device=latent.device,
                                                        dtype=torch.float32)
            self.observe_delta_translation = torch.zeros(bs, self.observe_frame_number, 2, device=latent.device,
                                                         dtype=torch.float32)
            self.observe_ego_lcf_feat = gt_ego_lcf_feat[:, 0:1].repeat(1, self.observe_frame_number, 1)
            self.observe_ego_mode = gt_ego_fut_cmd[:, 0:1].repeat(1, self.observe_frame_number, 1)
            self.observe_curr_to_futu = torch.zeros(bs, self.observe_frame_number, 4, 4, device=latent.device,
                                                    dtype=torch.float32)
            self.observe_plan_embed = torch.zeros(bs, self.observe_frame_number, 128, device=latent.device,
                                                  dtype=torch.float32)

        if start_of_sequence.sum() > 0:
            self.observe_rotation[start_of_sequence] = ego_to_global_rotation[start_of_sequence, 0:1].repeat(1,
                                                                                                             self.observe_frame_number,
                                                                                                             1)
            self.observe_relative_rotation[start_of_sequence] = torch.ones(start_of_sequence.sum(),
                                                                           self.observe_frame_number, 4,
                                                                           device=latent.device, dtype=torch.float32)
            self.observe_delta_translation[start_of_sequence] = torch.zeros(start_of_sequence.sum(),
                                                                            self.observe_frame_number, 2,
                                                                            device=latent.device, dtype=torch.float32)
            self.observe_ego_lcf_feat[start_of_sequence] = gt_ego_lcf_feat[start_of_sequence, 0:1].repeat(1,
                                                                                                          self.observe_frame_number,
                                                                                                          1)
            self.observe_ego_mode[start_of_sequence] = gt_ego_fut_cmd[start_of_sequence, 0:1].repeat(1,
                                                                                                     self.observe_frame_number,
                                                                                                     1)
            self.observe_curr_to_futu[start_of_sequence] = torch.zeros(start_of_sequence.sum(),
                                                                       self.observe_frame_number, 4, 4,
                                                                       device=latent.device, dtype=torch.float32)
            self.observe_plan_embed[start_of_sequence] = torch.zeros(start_of_sequence.sum(), self.observe_frame_number,
                                                                     128, device=latent.device, dtype=torch.float32)

        # Update observe information
        self.observe_ego_mode = torch.cat([self.observe_ego_mode[:, 1:], gt_ego_fut_cmd[:, 0:1]], dim=1)
        self.observe_rotation = torch.cat([self.observe_rotation[:, 1:], ego_to_global_rotation[:, 0:1]], dim=1)
        self.observe_ego_lcf_feat = torch.cat([self.observe_ego_lcf_feat[:, 1:], gt_ego_lcf_feat[:, 0:1]], dim=1)

        # -------------- Init hisotry & input tinformation --------------
        history_token = latent[:, 0:1].repeat(1, self.memory_frame_number, 1, 1, 1).detach().clone()  # bs, f, c, w, h
        history_rotation = ego_to_global_rotation[:, 0:1].repeat(1, self.memory_frame_number, 1).detach().clone()
        history_relative_rotation = torch.ones(bs, self.memory_frame_number, 4, device=latent.device,
                                               dtype=torch.float32)
        history_delta_translation = torch.zeros(bs, self.memory_frame_number, 2, device=latent.device,
                                                dtype=torch.float32)
        history_ego_lcf_feat = gt_ego_lcf_feat[:, 0:1].repeat(1, self.memory_frame_number, 1).detach().clone()
        history_ego_mode = gt_ego_fut_cmd[:, 0:1].repeat(1, self.memory_frame_number, 1).detach().clone()
        history_curr_to_future = torch.zeros(bs, self.memory_frame_number, 4, 4, device=latent.device,
                                             dtype=torch.float32)
        history_plan_embed = torch.zeros(bs, self.memory_frame_number, 128, device=latent.device, dtype=torch.float32)

        # update history with observe information
        history_rotation[:, -self.observe_frame_number:] = self.observe_rotation
        history_relative_rotation[:, -self.observe_frame_number:] = self.observe_relative_rotation
        history_delta_translation[:, -self.observe_frame_number:] = self.observe_delta_translation
        history_ego_lcf_feat[:, -self.observe_frame_number:] = self.observe_ego_lcf_feat
        history_ego_mode[:, -self.observe_frame_number:] = self.observe_ego_mode
        history_curr_to_future[:, -self.observe_frame_number:] = self.observe_curr_to_futu
        history_plan_embed[:, -self.observe_frame_number:] = self.observe_plan_embed

        # Init the current latent
        curr_latent = latent[:, 0].detach().clone()
        curr_curr_to_futu = curr_to_future_ego_rt[:, 0].detach().clone()
        curr_ego_to_global = curr_ego_to_global_rt[:, 0].detach().clone()
        curr_rotation = ego_to_global_rotation[:, 0].detach().clone()
        curr_relative_rotation = torch.ones(bs, 4, device=latent.device, dtype=torch.float32)
        curr_translation = ego_to_global_translation[:, 0].detach().clone()
        curr_ego_lcf_feat = gt_ego_lcf_feat[:, 0].detach().clone()
        curr_ego_mode = gt_ego_fut_cmd[:, 0].detach().clone()
        curr_delta_translation = torch.zeros(bs, 2, device=latent.device, dtype=torch.float32)
        # Autogressive prediction
        pred_latents, pred_rotations, pred_delta_translations, pred_ego_to_globals = [], [], [], []
        pred_relative_rotations = []
        pred_plan_embeds = []
        pred_ego_modes = []
        next_rotation = ego_to_global_rotation[:, 0]

        for frame_idx in range(predict_future_frame):
            # Decide whether to use GT
            use_gt_rate = torch.rand(size=(bs,), device=curr_latent.device) < self.sample_rate
            plan_query = self.pose_encoder.forward_encoder(
                history_delta_translation=history_delta_translation,
                history_ego_mode=history_ego_mode,
                history_ego_lcf_feat=history_ego_lcf_feat,
                history_relative_rotation=history_relative_rotation,
            )

            pred_latent, plan_embed, pred_delta_translation, pred_next_translation, pred_relative_rotation, pred_next_rotation, pred_next_ego_to_global = self.transformer(
                bev_queries=curr_latent,
                plan_queries=plan_query,
                hist_queries=history_token,
                curr_to_future_ego_rt=curr_to_future_ego_rt[:, frame_idx],
                curr_ego_to_global=curr_ego_to_global,
                curr_translation=curr_translation,
                curr_rotation=curr_rotation,
                next_rotation=ego_to_global_rotation[:, frame_idx + 1],
                use_gt_rate=use_gt_rate,
                use_gt=use_gt,
                train=train,
                history_plan_embed=history_plan_embed,
                curr_ego_mode=curr_ego_mode,
            )
            pred_ego_lcf_feat = self.pose_encoder.get_ego_feat(
                pred_rotation=pred_next_rotation,
                pred_translation=pred_next_translation,
                curr_rotation=curr_rotation,
                curr_translations=curr_translation,
                start_of_sequence=start_of_sequence,
            )

            # update information
            if use_gt:
                # Sample from the distribution
                curr_latent = torch.zeros_like(curr_latent)
                curr_latent[use_gt_rate] = latent[use_gt_rate, frame_idx + 1]
                curr_latent[~use_gt_rate] = pred_latent[~use_gt_rate]

                curr_delta_translation = torch.zeros_like(curr_delta_translation)
                curr_delta_translation[use_gt_rate] = ego_to_global_delta_translation[use_gt_rate, frame_idx]
                curr_delta_translation[~use_gt_rate] = pred_delta_translation[~use_gt_rate]

                curr_translation = torch.zeros_like(curr_translation)
                curr_translation[use_gt_rate] = ego_to_global_translation[use_gt_rate, frame_idx + 1]
                curr_translation[~use_gt_rate] = pred_next_translation[~use_gt_rate]

                curr_rotation = torch.zeros_like(curr_rotation)
                curr_rotation[use_gt_rate] = ego_to_global_rotation[use_gt_rate, frame_idx + 1]
                curr_rotation[~use_gt_rate] = pred_next_rotation[~use_gt_rate]

                curr_relative_rotation = torch.zeros_like(curr_relative_rotation)
                curr_relative_rotation[use_gt_rate] = ego_to_global_relative_rotation[use_gt_rate, frame_idx]
                curr_relative_rotation[~use_gt_rate] = pred_relative_rotation[~use_gt_rate]

                curr_ego_to_global = torch.zeros_like(curr_ego_to_global)
                curr_ego_to_global[use_gt_rate] = curr_ego_to_global_rt[use_gt_rate, frame_idx + 1]
                curr_ego_to_global[~use_gt_rate] = pred_next_ego_to_global[~use_gt_rate]

                curr_ego_lcf_feat = torch.zeros_like(curr_ego_lcf_feat)
                curr_ego_lcf_feat[use_gt_rate] = gt_ego_lcf_feat[use_gt_rate, frame_idx]
                curr_ego_lcf_feat[~use_gt_rate] = pred_ego_lcf_feat[~use_gt_rate]

                curr_ego_mode = gt_ego_fut_cmd[:, frame_idx]

            else:
                curr_latent = pred_latent.detach().clone()
                curr_delta_translation = pred_delta_translation.detach().clone()
                curr_translation = pred_next_translation.detach().clone()
                curr_rotation = pred_next_rotation.detach().clone()
                curr_relative_rotation = pred_relative_rotation.detach().clone()
                curr_ego_to_global = pred_next_ego_to_global.detach().clone()
                curr_ego_lcf_feat = pred_ego_lcf_feat.detach().clone()
                curr_ego_mode = gt_ego_fut_cmd[:, frame_idx]

            # Update history
            history_token = torch.cat([history_token[:, 1:], curr_latent.unsqueeze(1).detach().clone()], dim=1)
            history_delta_translation = torch.cat(
                [history_delta_translation[:, 1:], curr_delta_translation.unsqueeze(1).detach().clone()], dim=1)
            history_relative_rotation = torch.cat(
                [history_relative_rotation[:, 1:], curr_relative_rotation.unsqueeze(1).detach().clone()], dim=1)
            history_rotation = torch.cat([history_rotation[:, 1:], curr_rotation.unsqueeze(1).detach().clone()], dim=1)
            history_plan_embed = torch.cat([history_plan_embed[:, 1:], plan_embed.detach().clone()], dim=1)
            history_ego_lcf_feat = torch.cat(
                [history_ego_lcf_feat[:, 1:], curr_ego_lcf_feat.unsqueeze(1).detach().clone()], dim=1)
            history_ego_mode = torch.cat([history_ego_mode[:, 1:], curr_ego_mode.unsqueeze(1).detach().clone()], dim=1)

            # Store the intermediate results
            pred_latents.append(pred_latent)
            pred_delta_translations.append(pred_delta_translation)
            pred_plan_embeds.append(plan_embed)
            pred_ego_to_globals.append(pred_next_ego_to_global)
            pred_relative_rotations.append(pred_relative_rotation)

        # Update observe information
        self.observe_delta_translation = torch.cat(
            [self.observe_delta_translation[:, 1:], ego_to_global_delta_translation[:, 0:1]], dim=1)
        self.observe_relative_rotation = torch.cat(
            [self.observe_relative_rotation[:, 1:], ego_to_global_relative_rotation[:, 0:1]], dim=1)
        self.observe_plan_embed = torch.cat([self.observe_plan_embed[:, 1:], pred_plan_embeds[0].detach().clone()],
                                            dim=1)

        return_dict = dict(
            pred_latents=torch.stack(pred_latents, dim=1),  # [bs, f, c, w, h], pred future latents
            pred_delta_translations=torch.stack(pred_delta_translations, dim=1),
            # [bs, f, 2], pred future delta translations
            targ_delta_translations=ego_to_global_delta_translation,  # [bs, f, 2], GT future delta translations
            pred_relative_rotations=torch.stack(pred_relative_rotations, dim=1),  # [bs, f, 4], pred future rotations
            targ_relative_rotations=ego_to_global_relative_rotation,  # [bs, f, 4], GT future rotations
            pred_ego_to_globals=torch.stack(pred_ego_to_globals, dim=1),  # [bs, f, 4, 4], pred future ego to global
            targ_ego_to_globals=curr_ego_to_global_rt,  # [bs, f, 4, 4], GT future ego to global
        )
        if not train:
            return_dict.update(
                ego_from_sensor=ego_from_sensor,  # [bs, f, 4, 4], ego from sensor
                targ_pose_mat=targ_pose_mat,  # [bs, f, 4, 4], GT pose mat
                gt_bboxes_3d=gt_bboxes_3d,
                gt_attr_labels=gt_attr_labels,
                ego_to_lidar=ego_to_lidar,
            )
        return return_dict

    def forward_test(self, latent, voxel_semantics, img_metas, **kwargs):
        # Forward current latent
        pred_curr_voxel_semantics = self.obtain_scene_from_token(latent[:, 0])
        pred_curr_voxel_semantics = pred_curr_voxel_semantics.softmax(-1).argmax(-1)
        targ_future_voxel_semantics = voxel_semantics[:, self.test_previous_frame + 1:]
        targ_curr_voxel_semantics = voxel_semantics[:, self.test_previous_frame:self.test_previous_frame + 1]

        # Autoregressive predict future latent & Forward future latent
        return_dict = self.forward_sample(latent, img_metas, self.test_future_frame, use_gt=False, train=False)
        pred_latents = return_dict['pred_latents']
        pred_voxel_semantics = self.obtain_scene_from_token(pred_latents)
        pred_voxel_semantics = pred_voxel_semantics.softmax(-1).argmax(-1)

        vis_voxel_semantics = torch.cat([pred_curr_voxel_semantics, pred_voxel_semantics], dim=1)

        # Process Trajectory info
        ego_from_sensor, targ_pose_mat = return_dict['ego_from_sensor'], return_dict['targ_pose_mat']
        targ_ego_to_globals, pred_ego_to_global = return_dict['targ_ego_to_globals'], return_dict['pred_ego_to_globals']
        ego_to_lidar = return_dict['ego_to_lidar']
        pred_pose_mat = torch.matmul(pred_ego_to_global,
                                     ego_from_sensor.unsqueeze(1).repeat(1, self.test_future_frame, 1, 1))
        pred_pose_mat = torch.cat([targ_pose_mat[:, 0:1], pred_pose_mat], dim=1)
        pred_ego_to_global = torch.cat([targ_ego_to_globals[:, 0:1], pred_ego_to_global], dim=1)
        pred_trajs = self.decode_pose(pred_pose_mat, pred_ego_to_global, ego_to_lidar)  # pred
        # pred_trajs = self.decode_pose(targ_pose_mat, targ_ego_to_globals, ego_to_lidar)    # GT
        targ_trajs = np.array([img_meta['gt_ego_fut_trajs'] for img_meta in img_metas])

        #
        gt_bboxes_3d = return_dict['gt_bboxes_3d'].tensor.cpu().numpy()
        gt_attr_labels = return_dict['gt_attr_labels']

        return_dict = dict()
        sample_idx = img_metas[0]['sample_idx']
        # Occupancy prediction
        return_dict['pred_futu_semantics'] = pred_voxel_semantics.cpu().numpy().astype(np.uint8)
        return_dict['pred_curr_semantics'] = pred_curr_voxel_semantics.cpu().numpy().astype(np.uint8)
        return_dict['targ_futu_semantics'] = targ_future_voxel_semantics.cpu().numpy().astype(np.uint8)
        return_dict['targ_curr_semantics'] = targ_curr_voxel_semantics.cpu().numpy().astype(np.uint8)

        # Trajectory prediction
        return_dict['pred_ego_fut_trajs'] = pred_trajs.astype(np.float16)
        return_dict['targ_ego_fut_trajs'] = targ_trajs.astype(np.float16)

        return_dict['gt_bboxes_3d'] = gt_bboxes_3d
        return_dict['gt_attr_labels'] = gt_attr_labels

        # Other information
        return_dict['occ_index'] = [img_meta['occ_index'] for img_meta in img_metas]
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        return_dict['sample_idx'] = sample_idx

        return [return_dict]

    def forward_train(self, latent, img_metas, **kwargs):
        # Forward auto-regressive prediction
        return_dict = self.forward_sample(latent, img_metas, self.train_future_frame, use_gt=True, train=True)
        pred_latents = return_dict['pred_latents']
        targ_latents = latent[:, 1:]  # GT future latent

        pred_delta_translations = return_dict['pred_delta_translations']
        targ_delta_translations = return_dict['targ_delta_translations']

        pred_relative_rotations = return_dict['pred_relative_rotations']
        targ_relative_rotations = return_dict['targ_relative_rotations']

        # Get valid index for training
        valid_frame = torch.stack(
            [torch.tensor(img_meta['valid_frame'], device=latent.device) for img_meta in img_metas])

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
