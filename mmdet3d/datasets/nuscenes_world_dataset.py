# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

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
class NuScenesWorldDataset(Custom3DDataset):

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 test_mode=False,
                 load_future_frame_number=0,
                 load_previous_frame_number=0,
                 load_previous_data=False,
                 filter_empty_gt=False,
                 # SOLLOFusion
                 use_sequence_group_flag=False,
                 sequences_split_num=1,
                 # BEVAug
                 bda_aug_conf=None,
                 #
                 dataset_name='occ3d',
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
            filter_empty_gt=filter_empty_gt
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
        # BEVAug
        self.bda_aug_conf = bda_aug_conf
        #
        self.dataset_name = dataset_name
        self.eval_metric = eval_metric
        self.eval_time = eval_time
        self.box_mode_3d = Box3DMode.LIDAR

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """

        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['prev']) == 0:
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

    def compute_L2(self, trajs, gt_trajs):
        '''
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        '''
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
        # import pdb; pdb.set_trace()
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                np.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )

        return ade

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
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
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            can_bus=info['can_bus'],
            prev=info['prev'],
            scene_name=info['occ_path'].split('/')[-2]
        )

        if self.use_sequence_group_flag:
            input_dict['sample_index'] = index
            input_dict['sequence_group_idx'] = self.flag[index]
            input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]

            if not input_dict['start_of_sequence']:
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(
                    nuscenes_get_rt_matrix(self.data_infos[index], self.data_infos[index - 1], "ego", "ego"))
            else:
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(
                    nuscenes_get_rt_matrix(self.data_infos[index], self.data_infos[index], "ego", "ego"))

        occ_index = [index]

        # Load previous frame info
        input_dict.update(self.load_previous_frame_info(index, info))

        # Load future frame info
        input_dict.update(self.load_future_frame_info(index, info))

        # Update occ_index
        input_dict['occ_index'] = input_dict['previous_occ_index'] + occ_index + input_dict['future_occ_index']
        sample_weight = 1
        if len(input_dict['previous_occ_index']) != len(set(input_dict['previous_occ_index'])):
            sample_weight = 0.1
        input_dict['sample_weight'] = sample_weight
        input_dict['ego_from_sensor'] = info['ego_from_sensor']

        # Load Ego traj info
        input_dict.update(self.get_ego_trajs_info(index, info))

        # Load Box info
        input_dict.update(self.get_box_info(index))

        return input_dict

    def load_future_frame_info(self, index, info):
        future_occ_path = []  # list of future occupancy path
        future_occ_index = []  # list of future occupancy index
        curr_to_future_ego_rt = []  # list of transformation from current frame to future frame, calc by ego2global_t+1.inverse @ ego2global_t
        curr_ego_to_global_rt = []  # list of transformation from ego to global, store from current frame to future frame
        ego_to_global_rotation = []  # list of ego to global rotation, store the quaternion for training
        ego_to_global_translation = []  # list of ego to global translation, store the translation for training
        pose_mat = []  # list of pose matrix, the pose matrix obtain from ego2global and ego2sensor, only used current frame
        ego_from_sensor = []  # list of ego from sensor, the transformation from sensor to ego,
        ego_to_lidar = []  # list of ego to lidar, the transformation from ego to lidar

        last_future_info = copy.deepcopy(info)
        last_future_index = index

        # init current frame information
        ego_to_global = transform_matrix(info['ego2global_translation'],
                                         pyquaternion.Quaternion(info['ego2global_rotation']))
        curr_ego_to_global_rt.append(ego_to_global)
        ego_to_global_rotation.append(info['ego2global_rotation'])
        ego_to_global_translation.append(info['ego2global_translation'])
        pose_mat.append(info['pose_mat'])
        ego_from_sensor.append(info['ego_from_sensor'])
        ego2lidar = transform_matrix(info['lidar2ego_translation'], pyquaternion.Quaternion(info['lidar2ego_rotation']),
                                     inverse=True)
        ego_to_lidar.append(ego2lidar)

        valid_frame = np.ones(self.load_future_frame_number, dtype=np.bool)
        for i in range(self.load_future_frame_number):
            future_index = min(index + i + 1, len(self.data_infos) - 1)
            # check the future frame is in the same sequence
            future_info = self.data_infos[future_index]
            if future_info['prev'] == last_future_info['token']:  # check the future frame is in the same sequence
                future_occ_path.append(future_info['occ_path'])
                future_occ_index.append(future_index)
                # get the transformation from current frame to previous frame
                future_prev_info = self.data_infos[max(future_index - 1, 0)]

                if future_prev_info['token'] == future_info['prev']:
                    curr_to_future_ego = nuscenes_get_rt_matrix(future_prev_info, future_info, "ego", "ego")
                else:
                    curr_to_future_ego = nuscenes_get_rt_matrix(future_info, future_info, "ego", "ego")
                curr_to_future_ego_rt.append(curr_to_future_ego)

                # get ego_to_global
                ego_to_global = transform_matrix(future_info['ego2global_translation'],
                                                 pyquaternion.Quaternion(future_info['ego2global_rotation']))
                curr_ego_to_global_rt.append(ego_to_global)
                ego_to_global_rotation.append(future_info['ego2global_rotation'])
                ego_to_global_translation.append(future_info['ego2global_translation'])

                pose_mat.append(future_info['pose_mat'])

                # ego2lidar
                ego2lidar = transform_matrix(future_info['lidar2ego_translation'],
                                             pyquaternion.Quaternion(future_info['lidar2ego_rotation']), inverse=True)
                ego_to_lidar.append(ego2lidar)

                # update the last future info
                last_future_info = copy.deepcopy(future_info)
                last_future_index = copy.deepcopy(future_index)
            else:
                future_occ_path.append(last_future_info['occ_path'])
                future_occ_index.append(last_future_index)

                # get the transformation from current frame to previous frame
                curr_to_future_ego = nuscenes_get_rt_matrix(last_future_info, last_future_info, 'ego', 'ego')
                curr_to_future_ego_rt.append(curr_to_future_ego)

                # get ego_to_global
                ego_to_global = transform_matrix(last_future_info['ego2global_translation'],
                                                 pyquaternion.Quaternion(last_future_info['ego2global_rotation']))
                curr_ego_to_global_rt.append(ego_to_global)
                ego_to_global_rotation.append(last_future_info['ego2global_rotation'])
                ego_to_global_translation.append(last_future_info['ego2global_translation'])

                pose_mat.append(last_future_info['pose_mat'])

                # ego2lidar
                ego2lidar = transform_matrix(last_future_info['lidar2ego_translation'],
                                             pyquaternion.Quaternion(last_future_info['lidar2ego_rotation']),
                                             inverse=True)
                ego_to_lidar.append(ego2lidar)

                valid_frame[i:] = False

        output_dict = dict(
            future_occ_path=future_occ_path,
            future_occ_index=future_occ_index,
            curr_to_future_ego_rt=np.array(curr_to_future_ego_rt),
            curr_ego_to_global=np.array(curr_ego_to_global_rt),
            ego_to_global_rotation=np.array(ego_to_global_rotation),
            ego_to_global_translation=np.array(ego_to_global_translation),
            pose_mat=np.array(pose_mat),
            valid_frame=valid_frame,
            ego_to_lidar=np.array(ego_to_lidar)
        )
        return output_dict

    def load_previous_frame_info(self, index, info):
        previous_occ_path = []  # list of previous occupancy path
        previous_occ_index = []  # list of previous occupancy index
        previous_curr_to_prev_ego_rt = []  # list of transformation from current frame to previous frame
        last_previous_info = copy.deepcopy(info)
        last_previous_index = index
        for i in range(self.load_previous_frame_number):
            previous_index = max(index - i - 1, 0)
            # check the previous frame is in the same sequence
            previous_info = self.data_infos[previous_index]

            if previous_info['token'] == last_previous_info['prev']:
                previous_occ_index.append(previous_index)
                if self.load_previous_data:
                    previous_occ_path.append(previous_info['occ_path'])

                # get the transformation from current frame to previous frame
                prev_prev_info = self.data_infos[max(previous_index - 1, 0)]
                if prev_prev_info['token'] == previous_info['prev']:
                    curr_to_prev_ego_rt = nuscenes_get_rt_matrix(previous_info, prev_prev_info, "ego", "ego")
                else:
                    curr_to_prev_ego_rt = nuscenes_get_rt_matrix(previous_info, previous_info, "ego", "ego")

                previous_curr_to_prev_ego_rt.append(curr_to_prev_ego_rt)

                # update the last previous info
                last_previous_info = copy.deepcopy(previous_info)
                last_previous_index = copy.deepcopy(previous_index)
            else:
                if self.load_previous_data:
                    previous_occ_path.append(last_previous_info['occ_path'])

                # get the transformation from current frame to previous frame
                curr_to_prev_ego_rt = nuscenes_get_rt_matrix(last_previous_info, last_previous_info, "ego", "ego")
                previous_curr_to_prev_ego_rt.append(curr_to_prev_ego_rt)

                previous_occ_index.append(last_previous_index)

        output_dict = dict(
            previous_occ_path=list(reversed(previous_occ_path)),
            previous_curr_to_prev_ego_rt=list(reversed(previous_curr_to_prev_ego_rt)),
            previous_occ_index=list(reversed(previous_occ_index))
        )
        return output_dict

    def get_ego_trajs_info(self, index, info):
        # get traj info
        gt_ego_fut_trajs = []
        gt_ego_fut_cmd = []
        gt_ego_lcf_feat = []
        for i in range(self.load_future_frame_number):
            if i == 0:
                trajs = info['gt_ego_fut_trajs']
                gt_ego_fut_trajs.append(trajs[0])
                gt_ego_fut_cmd.append(info['gt_ego_fut_cmd'])
                ego_feat = info['gt_ego_lcf_feat']
                ego_vx, ego_vy, ego_w = ego_feat[0], ego_feat[1], ego_feat[4]
                gt_ego_lcf_feat.append([ego_vx, ego_vy, ego_w])
            else:
                get_info = self.data_infos[min(index + i, len(self.data_infos) - 1)]
                trajs = get_info['gt_ego_fut_trajs']
                gt_ego_fut_trajs.append(trajs[0])
                gt_ego_fut_cmd.append(get_info['gt_ego_fut_cmd'])
                ego_feat = get_info['gt_ego_lcf_feat']
                ego_vx, ego_vy, ego_w = ego_feat[0], ego_feat[1], ego_feat[4]
                gt_ego_lcf_feat.append([ego_vx, ego_vy, ego_w])

        gt_ego_fut_trajs = np.array(gt_ego_fut_trajs)
        gt_ego_fut_cmd = np.array(gt_ego_fut_cmd)
        gt_ego_lcf_feat = np.array(gt_ego_lcf_feat)
        output_dict = dict(
            gt_ego_fut_trajs=gt_ego_fut_trajs,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            gt_ego_lcf_feat=gt_ego_lcf_feat,
        )
        return output_dict

    def get_box_info(self, index):
        info = self.data_infos[index]
        # Load Box info
        mask = info['valid_flag']
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_velocity = info['gt_velocity'][mask]
        nan_mask = np.isnan(gt_velocity[:, 0])
        gt_velocity[nan_mask] = [0.0, 0.0]
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        gt_fut_trajs = info['gt_agent_fut_trajs'][mask]  # N, 2*6
        gt_fut_masks = info['gt_agent_fut_masks'][mask]  # N, 6
        gt_fut_goal = info['gt_agent_fut_goal'][mask]  # N
        gt_lcf_feat = info['gt_agent_lcf_feat'][mask]  # N, 9
        gt_fut_yaw = info['gt_agent_fut_yaw'][mask]  # N, 6
        attr_labels = np.concatenate(
            [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
        ).astype(np.float32)
        output_dict = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_names_3d=gt_names_3d,
            gt_attr_labels=attr_labels,
            fut_valid_flag=mask
        )

        return output_dict

    def evaluate_miou(self, results, logger=None):
        pred_sems, gt_sems = [], []
        data_index = []
        times = []

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
                time = result['time'] if 'time' in result else 0
                data_index.append(id)
                pred_sems.append(pred_sem)
                times.append(time)

        # calc the average time
        if len(times) > 0:
            avg_time = sum(times) / len(times)
            print(f'Average time: {avg_time}')

        for index in tqdm(data_index):
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]

            occ_path = info['occ_path']
            if self.dataset_name == 'openocc':
                occ_path = occ_path.replace('gts', 'openocc_v2')
            occ_path = os.path.join(occ_path, 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)

            gt_semantics = occ_gt['semantics']
            pr_semantics = pred_sems[data_index.index(index)]

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
        self.curr_miou_metric = Metric_mIoU(num_classes=num_classes, use_lidar_mask=False, use_image_mask=False,
                                            logger=logger)

        data_index = []
        occ_index = []
        # Occupancy-related
        pred_curr_sems, pred_futu_sems, targ_curr_sems, targ_futu_sems = [], [], [], []
        times = []

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
                time = result['time'] if 'time' in result else 0

                occ_index.append(result['occ_index'][i])
                data_index.append(id)
                pred_curr_sems.append(pred_curr_sem)
                pred_futu_sems.append(pred_futu_sem)
                targ_curr_sems.append(targ_curr_sem)
                targ_futu_sems.append(targ_futu_sem)
                times.append(time)

        # filter valid data
        # Occupancy-related
        valid_pred_curr_sems, valid_pred_futu_sems, valid_targ_curr_sems, valid_targ_futu_sems = [], [], [], []
        valid_times = []
        for i, occ_idx in tqdm(enumerate(occ_index)):
            if len(occ_idx) != len(set(occ_idx)):
                continue
            valid_pred_curr_sems.append(pred_curr_sems[i])
            valid_pred_futu_sems.append(pred_futu_sems[i])
            valid_targ_curr_sems.append(targ_curr_sems[i])
            valid_targ_futu_sems.append(targ_futu_sems[i])
            valid_times.append(times[i])

        # calc the average time
        if len(valid_times) > 0:
            avg_time = np.mean(valid_times)
            logger.info(f'Average time: {avg_time:.2f}s')

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