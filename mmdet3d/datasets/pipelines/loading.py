# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from configs.scene_tokenizer.ii_scene_tokenizer_waymo_4f import dataset_type
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES

from torchvision.transforms.functional import rotate

@PIPELINES.register_module()
class LoadStreamLatentToken(object):
    def __init__(self,
                 data_path=None,
                 to_long=False,
                 ):
        self.data_path = data_path
        self.to_long = to_long

    def __call__(self, results):
        # current frame
        sample_idx = results['sample_idx']
        scene_name = results['scene_name']
        latent_token_path = os.path.join(self.data_path, scene_name, f'{sample_idx}.npz')
        latent_token = np.load(latent_token_path)
        token = latent_token['token'][None]

        # future frame
        future_tokens = []
        future_frame = len(results['future_occ_path'])
        for frame_idx in range(future_frame):
            future_sample_idx = results['future_occ_path'][frame_idx].split('/')[-1]
            latent_token_path = os.path.join(self.data_path, scene_name, f'{future_sample_idx}.npz')
            latent_token = np.load(latent_token_path)
            future_token = latent_token['token']
            future_tokens.append(future_token)
        future_tokens = np.stack(future_tokens)

        input_token = np.concatenate([token, future_tokens], axis=0)

        results['latent'] = input_token
        return results

@PIPELINES.register_module()
class LoadLatentToken(object):
    def __init__(self,
                 data_path=None,
                 to_long=False,
                 ):
        self.data_path = data_path
        self.to_long = to_long

    def __call__(self, results):
        sample_idx = results['sample_idx']
        scene_name = results['scene_name']
        latent_token_path = os.path.join(self.data_path, scene_name, f'{sample_idx}.npz')
        latent_token = np.load(latent_token_path)
        token = latent_token['token']
        gt_mode = latent_token['gt_mode']
        rel_poses = latent_token['rel_poses']

        results['latent'] = token
        results['gt_mode'] = gt_mode
        results['rel_poses'] = rel_poses
        return results

@PIPELINES.register_module()
class LoadStreamOcc3D(object):
    def __init__(self,
                 to_long=True,
                 to_float=False,
                 dataset_type='occ3d',
                 ):
        self.to_long = to_long
        self.to_float = to_float
        self.dataset_type = dataset_type
        # waymo dataset cls map to occ3d
        self.waymo_map = {
            0: 0,  # TYPE_GENERALOBJECT
            1: 4,  # TYPE_VEHICLE
            2: 7,  # TYPE_PEDESTRIAN
            3: 15,  # TYPE_SIGN
            4: 2,  # TYPE_CYCLIST
            5: 15,  # TYPE_TRAFFIC_LIGHT
            6: 15,  # TYPE_POLE
            7: 8,  # TYPE_CONSTRUCTION_CONE
            8: 2,  # TYPE_BICYCLE
            9: 6,  # TYPE_MOTORCYCLE
            10: 15,  # TYPE_BUILDING
            11: 16,  # TYPE_VEGETATION
            12: 16,  # TYPE_TREE_TRUNK
            13: 11,  # TYPE_ROAD
            14: 13,  # TYPE_WALKABLE
            23: 17,  # TYPE_FREE
        }

    def __call__(self, results):
        curr_occ_path = results['occ_path']
        previous_occ_path = results['previous_occ_path']
        future_occ_path = results['future_occ_path']
        # load current frame
        if self.dataset_type == 'occ3d':
            occ_gt_label = os.path.join(curr_occ_path, "labels.npz")
            occ_labels = np.load(occ_gt_label)
            curr_semantics = occ_labels['semantics']
        elif self.dataset_type == 'waymo':
            occ_labels = np.load(curr_occ_path)
            curr_semantics = occ_labels['voxel_label']
            # map the waymo cls to occ3d cls
            map_semantics = copy.deepcopy(curr_semantics)
            for key in self.waymo_map.keys():
                map_semantics[curr_semantics == key] = self.waymo_map[key]
            curr_semantics = map_semantics
        elif self.dataset_type == 'stcocc':
            for index in range(len(previous_occ_path)):
                previous_occ_path[index] = previous_occ_path[index].replace('gts', 'stc-results')
            for index in range(len(future_occ_path)):
                future_occ_path[index] = future_occ_path[index].replace('gts', 'stc-results')
            curr_occ_path = curr_occ_path.replace('gts', 'stc-results')
            curr_semantics = np.load(curr_occ_path)['semantics']

        # load previous frame
        previous_semantics = []
        for path in previous_occ_path:
            previous_occ_gt_label = os.path.join(path, "labels.npz")
            previous_occ_label = np.load(previous_occ_gt_label)
            previous_semantic = previous_occ_label['semantics']
            previous_semantics.append(previous_semantic)

        # load future frame
        future_semantics = []
        for path in future_occ_path:
            future_occ_gt_label = os.path.join(path, "labels.npz")
            future_occ_label = np.load(future_occ_gt_label)
            future_semantic = future_occ_label['semantics']
            future_semantics.append(future_semantic)

        occ_semantics = previous_semantics + [curr_semantics] + future_semantics
        occ_semantics = np.array(occ_semantics)

        if self.to_long:
            occ_semantics = occ_semantics.astype(np.int64)
        if self.to_float:
            occ_semantics = occ_semantics.astype(np.float32)
        results['voxel_semantics'] = occ_semantics

        return results

@PIPELINES.register_module()
class BEVAugStream(object):
    def __init__(self, bda_aug_conf, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy, tran_bda):
        # get rotation matrix
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([
            [rot_cos, -rot_sin, 0],
            [rot_sin, rot_cos, 0],
            [0, 0, 1]])
        scale_mat = torch.Tensor([
            [scale_ratio, 0, 0],
            [0, scale_ratio, 0],
            [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
        )

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])

        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def voxel_transform(self, results, flip_dx, flip_dy, rotate_bda=None):
        if flip_dx:
            results['voxel_semantics'] = results['voxel_semantics'][:, ::-1,...].copy()

        if flip_dy:
            results['voxel_semantics'] = results['voxel_semantics'][:, :, ::-1,...].copy()

        return results

    def __call__(self, results):
        # sample bda augmentation
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()

        # get bda matrix
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)

        # do voxel transformation
        results = self.voxel_transform(results, flip_dx=flip_dx, flip_dy=flip_dy)
        results['bda_mat'] = bda_mat

        return results

@PIPELINES.register_module()
class LoadOccGTFromFileCVPR2023(object):
    def __init__(self,
                 scale_1_2=False,
                 scale_1_4=False,
                 scale_1_8=False,
                 load_mask=False,
                 load_flow=False,
                 flow_gt_path=None,
                 ignore_invisible=False,
                 to_long=False,
                 ):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.ignore_invisible = ignore_invisible
        self.load_mask = load_mask
        self.load_flow = load_flow
        self.flow_gt_path = flow_gt_path
        self.to_long = to_long

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")

        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        if self.load_mask:
            voxel_mask = occ_labels['mask_camera']
            results['voxel_mask_camera'] = voxel_mask.astype(bool)
            if self.ignore_invisible:
                semantics[voxel_mask==0] = 255
        results['voxel_semantics'] = semantics

        if self.load_flow:
            W, H, Z = semantics.shape[0], semantics.shape[1], semantics.shape[2]
            scene_token = occ_gt_path.split('/')[-1]
            sparse_flow_path = os.path.join(self.flow_gt_path, scene_token+'.bin')
            sparse_flow_idx_path = os.path.join(self.flow_gt_path, scene_token+'_idx.bin')
            occ_flow = np.zeros((W*H*Z, 2), dtype=np.float16)
            sparse_flow = np.fromfile(sparse_flow_path, dtype=np.float16).reshape(-1, 3)[:, :2]
            sparse_idx = np.fromfile(sparse_flow_idx_path, dtype=np.int32).reshape(-1)
            occ_flow[sparse_idx] = sparse_flow
            occ_flow = occ_flow.reshape(W, H, Z, 2)
            if self.ignore_invisible:
                occ_flow[voxel_mask==0] = 255
            results['voxel_flow'] = occ_flow

        if self.scale_1_2:
            occ_labels_1_2 = np.load(occ_gt_label_1_2)
            semantics_1_2 = occ_labels_1_2['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_2['mask_camera']
                if self.ignore_invisible:
                    semantics_1_2[voxel_mask==0] = 255
                results['voxel_mask_camera_1_2'] = voxel_mask
            results['voxel_semantics_1_2'] = semantics_1_2
        if self.scale_1_4:
            occ_labels_1_4 = np.load(occ_gt_label_1_4)
            semantics_1_4 = occ_labels_1_4['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_4['mask_camera']
                if self.ignore_invisible:
                    semantics_1_4[voxel_mask==0] = 255
                results['voxel_mask_camera_1_4'] = voxel_mask
            results['voxel_semantics_1_4'] = semantics_1_4

        if self.scale_1_8:
            occ_labels_1_8 = np.load(occ_gt_label_1_8)
            semantics_1_8 = occ_labels_1_8['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_8['mask_camera']
                if self.ignore_invisible:
                    semantics_1_8[voxel_mask==0] = 255
                results['voxel_mask_camera_1_8'] = voxel_mask
            results['voxel_semantics_1_8'] = semantics_1_8

        if self.to_long:
            results['voxel_semantics'] = results['voxel_semantics'].astype(np.int64)
            if self.scale_1_2:
                results['voxel_semantics_1_2'] = results['voxel_semantics_1_2'].astype(np.int64)
            if self.scale_1_4:
                results['voxel_semantics_1_4'] = results['voxel_semantics_1_4'].astype(np.int64)
            if self.scale_1_8:
                results['voxel_semantics_1_8'] = results['voxel_semantics_1_8'].astype(np.int64)

        return results

@PIPELINES.register_module()
class LoadOccGTFromFileOpenOcc(object):
    def __init__(self, scale_1_2=False, scale_1_4=False, scale_1_8=False, load_ray_mask=False):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.load_ray_mask = load_ray_mask

    def __call__(self, results):
        gts_occ_gt_path = results['occ_gt_path']

        occ_ray_mask_path = gts_occ_gt_path.replace('gts', 'openocc_v2_ray_mask')
        occ_ray_mask = os.path.join(occ_ray_mask_path, 'labels.npz')
        occ_ray_mask_1_2 = os.path.join(occ_ray_mask_path, 'labels_1_2.npz')
        occ_ray_mask_1_4 = os.path.join(occ_ray_mask_path, 'labels_1_4.npz')
        occ_ray_mask_1_8 = os.path.join(occ_ray_mask_path, 'labels_1_8.npz')

        occ_gt_path = gts_occ_gt_path.replace('gts', 'openocc_v2')
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")
        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        flow = occ_labels['flow']

        if self.scale_1_2:
            occ_labels_1_2 = np.load(occ_gt_label_1_2)
            semantics_1_2 = occ_labels_1_2['semantics']
            flow_1_2 = occ_labels_1_2['flow']
            results['voxel_semantics_1_2'] = semantics_1_2
            results['voxel_flow_1_2'] = flow_1_2
            if self.load_ray_mask:
                ray_mask_1_2 = np.load(occ_ray_mask_1_2)
                ray_mask_1_2 = ray_mask_1_2['ray_mask2']
                results['ray_mask_1_2'] = ray_mask_1_2
        if self.scale_1_4:
            occ_labels_1_4 = np.load(occ_gt_label_1_4)
            semantics_1_4 = occ_labels_1_4['semantics']
            flow_1_4 = occ_labels_1_4['flow']
            results['voxel_semantics_1_4'] = semantics_1_4
            results['voxel_flow_1_4'] = flow_1_4
            if self.load_ray_mask:
                ray_mask_1_4 = np.load(occ_ray_mask_1_4)
                ray_mask_1_4 = ray_mask_1_4['ray_mask2']
                results['ray_mask_1_4'] = ray_mask_1_4
        if self.scale_1_8:
            occ_labels_1_8 = np.load(occ_gt_label_1_8)
            semantics_1_8 = occ_labels_1_8['semantics']
            flow_1_8 = occ_labels_1_8['flow']
            results['voxel_semantics_1_8'] = semantics_1_8
            results['voxel_flow_1_8'] = flow_1_8
            if self.load_ray_mask:
                ray_mask_1_8 = np.load(occ_ray_mask_1_8)
                ray_mask_1_8 = ray_mask_1_8['ray_mask2']
                results['ray_mask_1_8'] = ray_mask_1_8

        if self.load_ray_mask:
            ray_mask = np.load(occ_ray_mask)
            ray_mask = ray_mask['ray_mask2']
            results['ray_mask'] = ray_mask

        results['voxel_semantics'] = semantics
        results['voxel_flows'] = flow

        return results


@PIPELINES.register_module()
class LoadAnnotations(object):

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        return results