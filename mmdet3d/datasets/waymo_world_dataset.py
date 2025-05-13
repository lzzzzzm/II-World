# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import pickle
import mmcv
import math
import cv2 as cv
import numpy as np
import torch
import pyquaternion
from tqdm import tqdm
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .utils import nuscenes_get_rt_matrix
from .occ_metrics import Metric_mIoU
from terminaltables import AsciiTable
from nuscenes.utils.geometry_utils import transform_matrix
from mmdet3d.utils import get_root_logger
from .plan_metrics import PlanningMetric

from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, Box3DMode


@DATASETS.register_module()
class WaymoWorldDataset(Custom3DDataset):

    def __init__(self,
                 ann_file,
                 pose_file=None,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 test_mode=False,
                 load_future_frame_number=0,
                 load_previous_frame_number=0,
                 load_previous_data=False,
                 filter_empty_gt=False,
                 pts_prefix='velodyne',
                 split=None,
                 # SOLLOFusion
                 use_sequence_group_flag=False,
                 sequences_split_num=1,
                 dataset_name='waymo',
                 eval_metric='miou',
                 eval_time=(0, 1, 2, 3),
                 **kwargs
                 ):
        self.load_interval = load_interval
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            pts_prefix=pts_prefix,
            split=split,
            pose_file=pose_file,
        )

        self.load_previous_data = load_previous_data
        self.load_previous_frame_number = load_previous_frame_number
        self.load_future_frame_number = load_future_frame_number
        # SOLOFusion
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        # sequences_split_num splits eacgh sequence into sequences_split_num parts.
        if self.test_mode:
            assert self.sequences_split_num == 1
        if self.use_sequence_group_flag:
            self._set_sequence_group_flag()  # Must be called after load_annotations b/c load_annotations does sorting.
        self.dataset_name = dataset_name
        self.eval_metric = eval_metric
        self.eval_time = eval_time
        self.plan_metric = PlanningMetric()
        self.box_mode_3d = Box3DMode.LIDAR

    def _get_pts_filename(self, idx):
        pts_filename = os.path.join(self.root_split, self.pts_prefix, f'{idx:07d}.bin')
        return pts_filename

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """

        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]['prev'] is None:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0,
                                   bin_counts[curr_flag],
                                   math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data, key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]

        # Set the sequence index and frame index
        for info in data_infos:
            sample_idx = info['image']['image_idx']
            scene_idx = sample_idx % 1000000 // 1000
            frame_idx = sample_idx % 1000000 % 1000
            info['scene_idx'] = scene_idx

            info['frame_idx'] = frame_idx
            info['sample_idx'] = sample_idx
            pts_filename = self._get_pts_filename(sample_idx)
            info['pts_filename'] = pts_filename

            # Get Occupancy path
            basename = os.path.basename(pts_filename)
            seq_name = basename[1:4]
            frame_name = basename[4:7]
            occ_path = os.path.join(self.root_split, seq_name,  '{}_04.npz'.format(frame_name))
            info['occ_path'] = occ_path

            # Get pose info
            pose_info = self.pose_info[scene_idx][frame_idx]
            info['ego2global'] = pose_info[0]['ego2global']
            info['global2ego'] = np.linalg.inv(pose_info[0]['ego2global'])

        # Set prev characteristics
        prev = None
        now_scene_idx = None
        for info in data_infos:
            if now_scene_idx is None or now_scene_idx == info['scene_idx']:
                now_scene_idx = info['scene_idx']
                info['prev'] = prev
                prev = info['frame_idx']
            elif now_scene_idx != info['scene_idx']:
                info['prev'] = None
                prev = info['frame_idx']
                now_scene_idx = info['scene_idx']

        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            index=index,
            occ_path=info['occ_path'],
            sample_idx=info['sample_idx'],
            timestamp=info['timestamp'] / 1e6,
            prev=info['prev'],
            scene_name=info['scene_idx']
        )

        if self.use_sequence_group_flag:
            input_dict['sample_index'] = index
            input_dict['sequence_group_idx'] = self.flag[index]
            input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]

            if not input_dict['start_of_sequence']:
                curr_ego_to_global = self.data_infos[index]['ego2global']
                prev_global_to_ego = self.data_infos[index - 1]['global2ego']
                curr_to_prev_ego_rt = curr_ego_to_global @ prev_global_to_ego
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(curr_to_prev_ego_rt)
            else:
                curr_ego_to_global = self.data_infos[index]['ego2global']
                prev_global_to_ego = self.data_infos[index]['global2ego']
                curr_to_prev_ego_rt = curr_ego_to_global @ prev_global_to_ego
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(curr_to_prev_ego_rt)

        occ_index = [index]
        input_dict['previous_occ_path'] = []
        input_dict['future_occ_path'] = []

        # Update occ_index
        input_dict['occ_index'] =  occ_index
        return input_dict


    def evaluate_miou(self, results, logger=None):
        pred_sems, gt_sems = [], []
        data_index = []

        num_classes = 17 if self.dataset_name == 'openocc' else 18
        self.miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=False,
            logger=logger
        )

        print('\nStarting Evaluation...')
        processed_set = set()
        for result in results:
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set: continue
                processed_set.add(id)

                pred_sem = result['semantics'][i]
                gt_sem = result['targ_semantics'][i]
                data_index.append(id)
                pred_sems.append(pred_sem)
                gt_sems.append(gt_sem)

        for index in tqdm(data_index):
            if index >= len(self.data_infos):
                break

            pr_semantics = pred_sems[data_index.index(index)]
            gt_semantics = gt_sems[data_index.index(index)]

            self.miou_metric.add_batch(pr_semantics, gt_semantics, None, None)
            self.miou_metric.add_iou_batch(pr_semantics, gt_semantics, None, None)

        _, miou, _, _, _ = self.miou_metric.count_miou()
        iou = self.miou_metric.count_iou()
        eval_dict = {
            'semantics_miou': miou,
            'binary_iou': iou
        }
        return eval_dict

    def evaluate(self, results, logger=None, runner=None, show_dir=None, **eval_kwargs):
        if self.eval_metric == 'miou':
            return self.evaluate_miou(results, logger=logger)