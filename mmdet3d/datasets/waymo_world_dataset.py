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
            occ_path = occ_path.replace('\\', '/')
            info['occ_path'] = occ_path

            # Get pose info
            pose_info = self.pose_info[scene_idx][frame_idx]
            info['ego2global'] = pose_info[0]['ego2global']
            info['global2ego'] = np.linalg.inv(pose_info[0]['ego2global'])

            ego2global_rotation = pyquaternion.Quaternion(matrix=pose_info[0]['ego2global'][:3, :3]).q
            ego2global_translation = pose_info[0]['ego2global'][:3, 3]
            info['ego2global_rotation'] = ego2global_rotation
            info['ego2global_translation'] = ego2global_translation


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

        # Load previous frame info
        input_dict.update(self.load_previous_frame_info(index, info))

        # Load future frame info
        input_dict.update(self.load_future_frame_info(index, info))

        # Update occ_index
        input_dict['occ_index'] =  occ_index
        return input_dict

    def load_future_frame_info(self, index, info):
        future_occ_path = []                # list of future occupancy path
        future_occ_index = []               # list of future occupancy index
        curr_to_future_ego_rt = []          # list of transformation from current frame to future frame, calc by ego2global_t+1.inverse @ ego2global_t
        curr_ego_to_global_rt = []          # list of transformation from ego to global, store from current frame to future frame
        ego_to_global_rotation = []         # list of ego to global rotation, store the quaternion for training
        ego_to_global_translation = []      # list of ego to global translation, store the translation for training


        last_future_info = copy.deepcopy(info)
        last_future_index = index

        # init current frame information
        curr_ego_to_global_rt.append(info['ego2global'])
        ego_to_global_rotation.append(info['ego2global_rotation'])
        ego_to_global_translation.append(info['ego2global_translation'])

        valid_frame = np.ones(self.load_future_frame_number, dtype=np.bool)
        for i in range(self.load_future_frame_number):
            future_index = min(index + i + 1, len(self.data_infos) - 1)
            # check the future frame is in the same sequence
            future_info = self.data_infos[future_index]
            if future_info['prev'] == last_future_info['frame_idx']:    # check the future frame is in the same sequence
                future_occ_path.append(future_info['occ_path'])
                future_occ_index.append(future_index)
                # get the transformation from current frame to previous frame
                future_prev_info = self.data_infos[max(future_index - 1, 0)]

                if future_prev_info['frame_idx'] == future_info['prev']:
                    curr_ego_to_global = future_prev_info['ego2global']
                    futu_global_to_ego = future_info['global2ego']
                    curr_to_future_ego = curr_ego_to_global @ futu_global_to_ego

                else:
                    curr_ego_to_global = future_info['ego2global']
                    futu_global_to_ego = future_info['global2ego']
                    curr_to_future_ego = curr_ego_to_global @ futu_global_to_ego
                curr_to_future_ego_rt.append(curr_to_future_ego)

                # get ego_to_global
                ego_to_global = future_info['ego2global']
                curr_ego_to_global_rt.append(ego_to_global)
                ego_to_global_rotation.append(future_info['ego2global_rotation'])
                ego_to_global_translation.append(future_info['ego2global_translation'])

                # update the last future info
                last_future_info = copy.deepcopy(future_info)
                last_future_index = copy.deepcopy(future_index)
            else:
                future_occ_path.append(last_future_info['occ_path'])
                future_occ_index.append(last_future_index)

                # get the transformation from current frame to previous frame
                curr_ego_to_global = last_future_info['ego2global']
                futu_global_to_ego = last_future_info['global2ego']
                curr_to_future_ego = curr_ego_to_global @ futu_global_to_ego
                curr_to_future_ego_rt.append(curr_to_future_ego)

                # get ego_to_global
                ego_to_global = last_future_info['ego2global']
                curr_ego_to_global_rt.append(ego_to_global)
                ego_to_global_rotation.append(last_future_info['ego2global_rotation'])
                ego_to_global_translation.append(last_future_info['ego2global_translation'])

                valid_frame[i:] = False

        output_dict = dict(
            future_occ_path=future_occ_path,
            future_occ_index=future_occ_index,
            curr_to_future_ego_rt=np.array(curr_to_future_ego_rt),
            curr_ego_to_global=np.array(curr_ego_to_global_rt),
            ego_to_global_rotation=np.array(ego_to_global_rotation),
            ego_to_global_translation=np.array(ego_to_global_translation),
            valid_frame=valid_frame,
        )
        return output_dict


    def load_previous_frame_info(self, index, info):
        previous_occ_path = []                      # list of previous occupancy path
        previous_occ_index = []                     # list of previous occupancy index
        last_previous_info = copy.deepcopy(info)
        last_previous_index = index
        for i in range(self.load_previous_frame_number):
            previous_index = max(index - i - 1, 0)
            # check the previous frame is in the same sequence
            previous_info = self.data_infos[previous_index]

            if previous_info['frame_idx'] == last_previous_info['prev']:
                previous_occ_index.append(previous_index)
                if self.load_previous_data:
                    previous_occ_path.append(previous_info['occ_path'])

                # get the transformation from current frame to previous frame
                prev_prev_info = self.data_infos[max(previous_index - 1, 0)]

                # update the last previous info
                last_previous_info = copy.deepcopy(previous_info)
                last_previous_index = copy.deepcopy(previous_index)
            else:
                if self.load_previous_data:
                    previous_occ_path.append(last_previous_info['occ_path'])

                previous_occ_index.append(last_previous_index)

        output_dict = dict(
            previous_occ_path=list(reversed(previous_occ_path)),
            previous_occ_index=list(reversed(previous_occ_index))
        )
        return output_dict

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

    def evaluate_forecasting_miou(self, results, logger):
        if logger is None:
            logger = get_root_logger()

        print('\nStarting Evaluation...')
        num_classes = 17 if self.dataset_name == 'openocc' else 18
        self.miou_metric_list = []
        for i in range(self.load_future_frame_number):
            self.miou_metric_list.append(
                Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=False, logger=logger)
            )
        self.curr_miou_metric = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=False, logger=logger)

        data_index = []
        occ_index = []
        # Occupancy-related
        pred_curr_sems, pred_futu_sems, targ_curr_sems, targ_futu_sems  = [], [], [], []

        processed_set = set()
        for result in results:
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set: continue
                processed_set.add(id)
                # Occupancy-related
                pred_curr_sem = result['pred_curr_semantics'][i]
                pred_futu_sem = result['pred_futu_semantics'][i]
                targ_curr_sem = result['targ_curr_semantics'][i]
                targ_futu_sem = result['targ_futu_semantics'][i]

                occ_index.append(result['occ_index'][i])
                data_index.append(id)
                pred_curr_sems.append(pred_curr_sem)
                pred_futu_sems.append(pred_futu_sem)
                targ_curr_sems.append(targ_curr_sem)
                targ_futu_sems.append(targ_futu_sem)

        # filter valid data
        # Occupancy-related
        valid_pred_curr_sems, valid_pred_futu_sems, valid_targ_curr_sems, valid_targ_futu_sems = [], [], [], []
        for i, occ_idx in tqdm(enumerate(occ_index)):
            if len(occ_idx) != len(set(occ_idx)):
                continue
            valid_pred_curr_sems.append(pred_curr_sems[i])
            valid_pred_futu_sems.append(pred_futu_sems[i])
            valid_targ_curr_sems.append(targ_curr_sems[i])
            valid_targ_futu_sems.append(targ_futu_sems[i])

        # delete invalid data
        pred_curr_sems = valid_pred_curr_sems
        pred_futu_sems = valid_pred_futu_sems
        targ_curr_sems = valid_targ_curr_sems
        targ_futu_sems = valid_targ_futu_sems

        # evaluate time 0s also means reconstructing the current frame
        eval_dict = dict()
        if 0 in self.eval_time:
            for i in tqdm(range(len(pred_curr_sems))):
                pred_curr_sem = pred_curr_sems[i][0]
                targ_curr_sem = targ_curr_sems[i][0]
                self.curr_miou_metric.add_batch(pred_curr_sem, targ_curr_sem, None, None)
                self.curr_miou_metric.add_iou_batch(pred_curr_sem, targ_curr_sem, None, None)
            print(f'evaluating time {0}s ----------------------')
            _, miou, _, _, _ = self.curr_miou_metric.count_miou()
            iou = self.curr_miou_metric.count_iou()
            eval_dict.update(
                {
                    f'semantics_miou_time_{0}s': miou,
                    f'binary_iou_time_{0}s': iou
                }
            )

        # evaluate future frames
        for i in tqdm(range(len(pred_futu_sems))):
            pred_futu_sem = pred_futu_sems[i]
            targ_futu_sem = targ_futu_sems[i]
            for j in range(pred_futu_sem.shape[0]):
                time = 0.5 * (j + 1)
                if time in self.eval_time:
                    self.miou_metric_list[j].add_batch(pred_futu_sem[j], targ_futu_sem[j], None, None)
                    self.miou_metric_list[j].add_iou_batch(pred_futu_sem[j], targ_futu_sem[j], None, None)

        # Create result tabel
        table_data = [['Time', 'mIoU', 'IoU']]

        restore_table_data = []
        for i in range(self.load_future_frame_number):
            time = 0.5 * (i + 1)
            if time in self.eval_time:
                _, miou, _, _, _ = self.miou_metric_list[i].count_miou()
                print(f'evaluating time {time}s ----------------------')
                iou = self.miou_metric_list[i].count_iou()
                restore_table_data.append([f'{time}s', f'{miou:.2f}', f'{iou:.2f}'])
                eval_dict.update(
                    {
                        f'semantics_miou_time_{time}s': miou,
                        f'binary_iou_time_{time}s': iou
                    }
                )
        # calc the average
        miou_list = []
        iou_list = []
        for i in range(self.load_future_frame_number):
            time = 0.5 * (i + 1)
            if time in self.eval_time:
                _, miou, _, _, _ = self.miou_metric_list[i].count_miou()
                iou = self.miou_metric_list[i].count_iou()
                miou_list.append(miou)
                iou_list.append(iou)
        restore_table_data.append(['Average', f'{np.mean(miou_list):.2f}', f'{np.mean(iou_list):.2f}'])

        for i in range(len(restore_table_data)):
            table_data.append(restore_table_data[i])

        table = AsciiTable(table_data)
        logger.info('Evaluation Results:')
        logger.info(table.table)
        return eval_dict

    def evaluate(self, results, logger=None, runner=None, show_dir=None, **eval_kwargs):
        if self.eval_metric == 'miou':
            return self.evaluate_miou(results, logger=logger)
        elif self.eval_metric == 'forecasting_miou':
            return self.evaluate_forecasting_miou(results, logger=logger)