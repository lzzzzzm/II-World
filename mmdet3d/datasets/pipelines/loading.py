# Copyright (c) OpenMMLab. All rights reserved.
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
        # gt_mode = latent_token['gt_mode']
        # rel_poses = latent_token['rel_poses']

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
        # results['gt_mode'] = gt_mode
        # results['rel_poses'] = rel_poses
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
                 ):
        self.to_long = to_long
        self.to_float = to_float

    def __call__(self, results):
        curr_occ_path = results['occ_path']
        previous_occ_path = results['previous_occ_path']
        future_occ_path = results['future_occ_path']
        # load current frame
        occ_gt_label = os.path.join(curr_occ_path, "labels.npz")
        occ_labels = np.load(occ_gt_label)
        curr_semantics = occ_labels['semantics']

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
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config
        self.index = 0
        self.num_cam = 6
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)

    def vis_depth_img(self, img, depth):
        depth = depth.cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * self.std + self.mean
        img = np.array(img, dtype=np.uint8)
        invalid_y, invalid_x, invalid_c = np.where(img == 0)
        depth[invalid_y, invalid_x] = 0
        y, x = np.where(depth != 0)
        plt.figure()
        plt.imshow(img)
        plt.scatter(x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2)
        plt.show()
        self.index = self.index + 1

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)

        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        # prev process info
        points_lidar = results['points'].tensor
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda = results['img_inputs']
        lidar2imgs = results['lidar2img']
        nt, c, h, w = imgs.shape
        t_frame = nt // 6

        # store list
        depth_maps = []                     # process result

        vis_index = 0
        for cid in range(len(results['cam_names'])):
            lidar2img = lidar2imgs[cid]

            # project lidar point to img plane
            points_img = lidar2img @ torch.cat([points_lidar.T, torch.ones((1, points_lidar.shape[0]))], dim=0)
            points_img = points_img.permute(1, 0)
            points_img = torch.cat([points_img[:, :2] / points_img[:, 2].unsqueeze(1), points_img[:, 2].unsqueeze(1)], dim=1)

            # get corresponding depth value
            depth_map = self.points2depthmap(points_img, h, w)

            # store
            depth_maps.append(depth_map)

            # vis depth img to check the correctness
            # self.vis_depth_img(imgs[cid*t_frame], depth_map)

        results['gt_depth'] = torch.stack(depth_maps)
        return results

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    to_rgb = True
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        opencv_pp=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.opencv_pp = opencv_pp

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        if not self.opencv_pp:
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        if self.opencv_pp:
            img = self.img_transform_core_opencv(img, post_rot, post_tran, crop)

        copy_img = img.copy()
        invalid_index = np.where(np.array(copy_img)==0)

        return img, post_rot, post_tran, invalid_index

    def img_transform_core_opencv(self, img, post_rot, post_tran,
                                  crop):
        img = np.array(img).astype(np.float32)
        img = cv2.warpAffine(img,
                             np.concatenate([post_rot,
                                            post_tran.reshape(2,1)],
                                            axis=1),
                             (crop[2]-crop[0], crop[3]-crop[1]),
                             flags=cv2.INTER_LINEAR)
        return img

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            random_crop_height = self.data_config.get('random_crop_height', False)
            if random_crop_height:
                crop_h = int(np.random.uniform(max(0.3*newH, newH-fH), newH-fH))
            else:
                crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
            if self.data_config.get('vflip', False) and np.random.choice([0, 1]):
                rotate += 180
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        # get sensor2ego
        sensor2ego = transform_matrix(
            translation=cam_info['cams'][cam_name]['sensor2ego_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['sensor2ego_rotation'])
        )
        sensor2ego = torch.from_numpy(sensor2ego).to(torch.float32)

        ego2sensor = transform_matrix(
            translation=cam_info['cams'][cam_name]['sensor2ego_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['sensor2ego_rotation']),
            inverse=True
        )
        ego2sensor = torch.from_numpy(ego2sensor).to(torch.float32)

        # get sensorego2global
        ego2global = transform_matrix(
            translation=cam_info['cams'][cam_name]['ego2global_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['ego2global_rotation'])
        )
        ego2global = torch.from_numpy(ego2global).to(torch.float32)

        global2ego = transform_matrix(
            translation=cam_info['cams'][cam_name]['ego2global_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['ego2global_rotation']),
            inverse=True
        )
        global2ego = torch.from_numpy(global2ego).to(torch.float32)

        return sensor2ego, ego2global, ego2sensor, global2ego

    def get_lidar_transformation(self, results):
        # get lidar2ego
        lidar2lidarego = transform_matrix(
            translation=results['curr']['lidar2ego_translation'],
            rotation=Quaternion(results['curr']['lidar2ego_rotation']),
        )
        lidar2lidarego = torch.from_numpy(lidar2lidarego).to(torch.float32)

        # get ego2lidar
        lidarego2lidar = transform_matrix(
            translation=results['curr']['lidar2ego_translation'],
            rotation=Quaternion(results['curr']['lidar2ego_rotation']),
            inverse=True
        )
        lidarego2lidar = torch.from_numpy(lidarego2lidar).to(torch.float32)

        # get ego2global
        lidarego2global = transform_matrix(
            translation=results['curr']['ego2global_translation'],
            rotation=Quaternion(results['curr']['ego2global_rotation']),
        )
        lidarego2global = torch.from_numpy(lidarego2global).to(torch.float32)

        return lidar2lidarego, lidarego2lidar, lidarego2global

    def photo_metric_distortion(self, img, pmd):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand()>pmd.get('rate', 1.0):
            return img

        img = np.array(img).astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-pmd['brightness_delta'],
                                   pmd['brightness_delta'])
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(pmd['saturation_lower'],
                                          pmd['saturation_upper'])

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-pmd['hue_delta'], pmd['hue_delta'])
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return Image.fromarray(img.astype(np.uint8))

    def get_inputs(self, results, flip=None, scale=None):
        # get cam_names
        cam_names = self.data_config['cams']

        # get store list
        imgs = []
        (sensor2egos, ego2globals, ego2sensors, global2egos, cam2imgs, post_augs,
         lidar2imgs, ego2lidars)  =\
            [], [], [], [], [], [], [], []

        # get lidar-related transformation
        lidar2lidarego, lidarego2lidar, lidarego2global = self.get_lidar_transformation(results)

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # get cam-related transformation
            cam2img = torch.eye(4)
            cam2img[:3, :3] = torch.tensor(cam_data['cam_intrinsic'][:3, :3], dtype=torch.float32)
            sensor2ego, ego2global, ego2sensor, global2ego = self.get_sensor_transforms(results['curr'], cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2, invalid_index = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 4x4
            post_aug = torch.eye(4)
            post_aug[:2, :2] = post_rot2
            post_aug[:2, 2] = post_tran2

            # get lidar2img
            lidar2img = cam2img @ ego2sensor @ global2ego @ lidarego2global @ lidar2lidarego
            lidar2img = post_aug @ lidar2img

            if self.is_train and self.data_config.get('pmd', None) is not None:
                img = self.photo_metric_distortion(img, self.data_config['pmd'])

            imgs.append(self.normalize_img(img))

            # adjacent frame use the same aug with current frame
            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    if self.opencv_pp:
                        img_adjacent = \
                            self.img_transform_core_opencv(
                                img_adjacent,
                                post_rot[:2, :2],
                                post_tran[:2],
                                crop)
                    else:
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            cam2imgs.append(cam2img)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            ego2sensors.append(ego2sensor)
            global2egos.append(global2ego)
            post_augs.append(post_aug)
            lidar2imgs.append(lidar2img)
        ego2lidars.append(lidarego2lidar)

        if self.sequential:
            for adj_info in results['adjacent']:
                # for convenience
                cam2imgs.extend(cam2imgs[:len(cam_names)])
                post_augs.extend(post_augs[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global, ego2sensor, global2ego = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)
                    ego2sensors.append(ego2sensor)
                    global2egos.append(global2ego)

        imgs = torch.stack(imgs)
        # sensor2egos and ego2globals containes current and adjacent frame information
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        ego2sensors = torch.stack(ego2sensors)
        global2egos = torch.stack(global2egos)
        # cam2imgs and post_augs only contain current frame information
        cam2imgs = torch.stack(cam2imgs)
        post_augs = torch.stack(post_augs)
        # lidar2imgs and ego2lidars only contain current frame information
        lidar2imgs = torch.stack(lidar2imgs)
        ego2lidars = torch.stack(ego2lidars)

        # store
        results['cam_names'] = cam_names
        results['sensor2sensorego'] = sensor2egos
        results['sensorego2global'] = ego2globals
        results['sensorego2sensor'] = ego2sensors
        results['global2sensorego'] = global2egos
        results['lidar2img'] = lidar2imgs
        results['ego2lidar'] = ego2lidars

        return (imgs, sensor2egos, ego2globals, cam2imgs, post_augs)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
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


@PIPELINES.register_module()
class BEVAug(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

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

    def voxel_transform(self, results, flip_dx, flip_dy):
        if flip_dx:
            results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
            if 'voxel_semantics_1_2' in results:
                results['voxel_semantics_1_2'] = results['voxel_semantics_1_2'][::-1, ...].copy()
            if 'voxel_semantics_1_4' in results:
                results['voxel_semantics_1_4'] = results['voxel_semantics_1_4'][::-1, ...].copy()
            if 'voxel_semantics_1_8' in results:
                results['voxel_semantics_1_8'] = results['voxel_semantics_1_8'][::-1, ...].copy()
            if 'voxel_flows' in results:
                results['voxel_flows'] = results['voxel_flows'][::-1, ...].copy()
                results['voxel_flows'][..., 0] = - results['voxel_flows'][..., 0]
                results['voxel_flows'][..., 0][results['voxel_flows'][..., 0] == -255] = 255

        if flip_dy:
            results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
            if 'voxel_semantics_1_2' in results:
                results['voxel_semantics_1_2'] = results['voxel_semantics_1_2'][:, ::-1, ...].copy()
            if 'voxel_semantics_1_4' in results:
                results['voxel_semantics_1_4'] = results['voxel_semantics_1_4'][:, ::-1, ...].copy()
            if 'voxel_semantics_1_8' in results:
                results['voxel_semantics_1_8'] = results['voxel_semantics_1_8'][:, ::-1, ...].copy()
            if 'voxel_flows' in results:
                results['voxel_flows'] = results['voxel_flows'][:, ::-1, ...].copy()
                results['voxel_flows'][..., 1] = - results['voxel_flows'][..., 1]
                results['voxel_flows'][..., 1][results['voxel_flows'][..., 1] == -255] = 255

        return results

    def __call__(self, results):
        # sample bda augmentation
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()
        if 'bda_aug' in results:
            flip_dx, flip_dy = results['bda_aug']['flip_dx'], results['bda_aug']['flip_dy']

        # get bda matrix
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)

        # do voxel transformation
        results = self.voxel_transform(results, flip_dx=flip_dx, flip_dy=flip_dy)

        # update img_inputs
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs = results['img_inputs']
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda_mat)

        return results
