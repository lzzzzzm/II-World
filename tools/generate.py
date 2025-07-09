# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
# warnings.filterwarnings('ignore')
import mmcv
import numpy as np
import torch
import cv2 as cv
import tqdm
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.models.utils import change_occupancy_to_bev
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import copy

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--scene_checkpoints',
        default='ckpts/ii_scene_tokenizer_4f.pth'
    )
    parser.add_argument(
        '--generate_path',
        default='generate_output'
    )
    parser.add_argument(
        '--generate_scene_name',
        default='scene-0564'
    )
    parser.add_argument(
        '--generate_frame',
        default=12,
        type=int,
    )
    parser.add_argument(
        '--task_mode',
        default='generate',
        type=str,
    )
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def show_translation_vector(save_path, curr_to_future_ego_rts, curr_to_future_ego_rts_sam, curr_to_future_ego_rts_sam2):
    """
    save the translation vector visualization
    """
    start_origin = np.zeros((2,))
    start_origin_sam = np.zeros((2,))
    start_origin_sam2 = np.zeros((2,))
    translation = curr_to_future_ego_rts[:, :2, 3]
    translation_sam = curr_to_future_ego_rts_sam[:, :2, 3]
    translation_sam2 = curr_to_future_ego_rts_sam2[:, :2, 3]
    print('translation_sam2 shape:', translation_sam2.shape)

    # subplot
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

    for j in range(0, translation.shape[0], 2):
        end = translation[j] + start_origin
        axs[0].quiver(start_origin[0], start_origin[1],
                   end[0] - start_origin[0], end[1] - start_origin[1],
                   angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)
        start_origin = end

    for j in range(0, translation_sam.shape[0], 2):
        end_sam = translation_sam[j] + start_origin_sam
        axs[1].quiver(start_origin_sam[0], start_origin_sam[1],
                   end_sam[0] - start_origin_sam[0], end_sam[1] - start_origin_sam[1],
                   angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)
        start_origin_sam = end_sam

    for j in range(0, translation_sam2.shape[0], 2):
        end_sam2 = translation_sam2[j] + start_origin_sam2
        axs[2].quiver(start_origin_sam2[0], start_origin_sam2[1],
                   end_sam2[0] - start_origin_sam2[0], end_sam2[1] - start_origin_sam2[1],
                   angles='xy', scale_units='xy', scale=1, color='b', alpha=0.5)
        start_origin_sam2 = end_sam2
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path.replace('bev', 'vector'), dpi=300)
    plt.close()


def show_high_level_forecast_bev(save_path, vis_bevs_right, vis_bevs_lefts, vis_bevs_straights):
    num_cols = 6
    num_rows = 6
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6))

    all_bevs = vis_bevs_right + vis_bevs_lefts + vis_bevs_straights
    time_list = [0.5 * (i+1) for i in range(len(vis_bevs_right))]
    all_titles = (['T:{}'.format(t) for t in time_list] * 3)

    for idx, (bev, title) in enumerate(zip(all_bevs, all_titles)):
        row = idx // num_cols
        col = idx % num_cols
        bev = cv.cvtColor(bev, cv.COLOR_RGB2BGR)
        axs[row, col].imshow(bev)
        axs[row, col].set_title(title)
        axs[row, col].axis('off')

    group_labels = ['Right', 'Left', 'Straight']
    group_centers = [1 / 6, 0.5, 5 / 6]

    for label, y in zip(group_labels, group_centers):
        fig.text(0.05, 1 - y, label, va='center', ha='center', fontsize=14, fontweight='bold')

    plt.subplots_adjust(left=0.1)

    for y in [2, 4]:
        fig.subplots_adjust(hspace=0.3)
        fig.lines.append(plt.Line2D([0, 1], [1 - y / num_rows, 1 - y / num_rows], color='black', linewidth=2,
                                    transform=fig.transFigure))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
    plt.close(fig)


def show_fine_forecast_bev(save_path, vis_bevs_ori, vis_bev_sams, vis_bev_sams_2, data, copy_data, copy_data_2):
    num_cols = 6
    num_rows = 6
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6))

    all_bevs = vis_bevs_ori + vis_bev_sams + vis_bev_sams_2
    time_list = [0.5 * (i+1) for i in range(len(vis_bevs_ori))]
    all_titles = (['T:{}'.format(t) for t in time_list] * 3)

    for idx, (bev, title) in enumerate(zip(all_bevs, all_titles)):
        row = idx // num_cols
        col = idx % num_cols
        bev = cv.cvtColor(bev, cv.COLOR_RGB2BGR)
        axs[row, col].imshow(bev)
        axs[row, col].set_title(title)
        axs[row, col].axis('off')
        # plot the ego-car points in (100, 100) in the BEV, utilize a blue point to represent the ego-car
        axs[row, col].plot(100, 100, 'bo', markersize=2)  # blue point at (100, 100)

    show_translation_vector(
        save_path,
        data['img_metas'].data[0][0]['curr_to_future_ego_rt'],
        copy_data['img_metas'].data[0][0]['curr_to_future_ego_rt'],
        copy_data_2['img_metas'].data[0][0]['curr_to_future_ego_rt']
    )

    group_labels = ['Ori', 'Sample1', 'Sample2']
    group_centers = [1 / 6, 0.5, 5 / 6]

    for label, y in zip(group_labels, group_centers):
        fig.text(0.05, 1 - y, label, va='center', ha='center', fontsize=14, fontweight='bold')

    plt.subplots_adjust(left=0.1)

    for y in [2, 4]:
        fig.subplots_adjust(hspace=0.3)
        fig.lines.append(plt.Line2D([0, 1], [1 - y / num_rows, 1 - y / num_rows], color='black', linewidth=2,
                                    transform=fig.transFigure))

    plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.close(fig)
    plt.show()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    samples_per_gpu = 1
    test_dataloader_default_args = dict(
        samples_per_gpu=samples_per_gpu, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    cfg.data.test.generate_mode = True
    cfg.data.test.generate_scene = args.generate_scene_name
    cfg.data.test.load_future_frame_number = args.generate_frame
    cfg.model.test_future_frame = args.generate_frame
    cfg.model.transformer.task_mode = args.task_mode
    dataset = build_dataset(cfg.data.test)
    dataset.generate_mode=True
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    if 'test_mode' in cfg.model:
        cfg.model.test_mode = True

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)


    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()
    save_root = os.path.join(args.generate_path, args.generate_scene_name)
    mmcv.mkdir_or_exist(save_root)

    curr_to_future_ego_rts = np.load('tools/curr_to_future_ego_rts.npz', allow_pickle=True)['curr_to_future_ego_rts']

    if args.task_mode == 'generate':
        np.random.seed(1)
        for i, data in enumerate(data_loader):
            # Random sample the future ego-rt
            sample_index_1 = np.random.randint(0, len(curr_to_future_ego_rts))
            sample_index_2 = np.random.randint(0, len(curr_to_future_ego_rts))

            copy_data = copy.deepcopy(data)
            copy_data_2 = copy.deepcopy(data)
            copy_data['img_metas'].data[0][0]['curr_to_future_ego_rt'] = curr_to_future_ego_rts[sample_index_1]
            copy_data_2['img_metas'].data[0][0]['curr_to_future_ego_rt'] = curr_to_future_ego_rts[sample_index_2]

            result_ori = model(return_loss=False, rescale=True, **data)
            result_sam = model(return_loss=False, rescale=True, **copy_data)
            result_sam_2 = model(return_loss=False, rescale=True, **copy_data_2)
            pred_semantics_result_ori = result_ori[0]['pred_futu_semantics'][0]
            pred_semantics_result_sam = result_sam[0]['pred_futu_semantics'][0]
            pred_semantics_result_sam_2 = result_sam_2[0]['pred_futu_semantics'][0]
            vis_bev_oris, vis_bev_sams, vis_bev_sams_2 = [], [], []
            for semantics_ori, semantics_sam, semantics_sam_2 in zip(pred_semantics_result_ori, pred_semantics_result_sam, pred_semantics_result_sam_2):
                vis_bev_ori = change_occupancy_to_bev(semantics_ori)
                vis_bev_sam = change_occupancy_to_bev(semantics_sam)
                vis_bev_sam_2 = change_occupancy_to_bev(semantics_sam_2)
                vis_bev_sam_2 = cv.cvtColor(vis_bev_sam_2, cv.COLOR_RGB2BGR)
                vis_bev_oris.append(vis_bev_ori)
                vis_bev_sams.append(vis_bev_sam)
                vis_bev_sams_2.append(vis_bev_sam_2)
            save_path = os.path.join(os.path.join(save_root, '{}'.format(i)))
            show_fine_forecast_bev(save_path, vis_bev_oris, vis_bev_sams, vis_bev_sams_2, data, copy_data, copy_data_2)
    elif args.task_mode == 'high_level_control':
        # Generate example of high-level control, utilize the cmd state to control the future forecast
        for i, data in enumerate(data_loader):
            print('Number of data: {}'.format(i))
            with torch.no_grad():
                # Simply change to cmd to
                control_data_right = copy.deepcopy(data)
                control_data_left = copy.deepcopy(data)
                control_data_straight = copy.deepcopy(data)
                control_data_right['img_metas'].data[0][0]['gt_ego_fut_cmd'] = np.zeros((args.generate_frame, 3))
                control_data_right['img_metas'].data[0][0]['gt_ego_fut_cmd'][..., 0] = 1
                control_data_left['img_metas'].data[0][0]['gt_ego_fut_cmd'] = np.zeros((args.generate_frame, 3))
                control_data_left['img_metas'].data[0][0]['gt_ego_fut_cmd'][..., 1] = 1
                control_data_straight['img_metas'].data[0][0]['gt_ego_fut_cmd'] = np.zeros((args.generate_frame, 3))
                control_data_straight['img_metas'].data[0][0]['gt_ego_fut_cmd'][..., 2] = 1

                result_right = model(return_loss=False, rescale=True, **control_data_right)
                result_left = model(return_loss=False, rescale=True, **control_data_left)
                result_straight = model(return_loss=False, rescale=True, **control_data_straight)

                pred_semantics_right = result_right[0]['pred_futu_semantics'][0]
                pred_semantics_left = result_left[0]['pred_futu_semantics'][0]
                pred_semantics_straight = result_straight[0]['pred_futu_semantics'][0]
                vis_bevs_rights = []
                vis_bevs_lefts = []
                vis_bevs_straights = []
                for semantics_right, semantics_left, semantics_straight in zip(pred_semantics_right, pred_semantics_left, pred_semantics_straight):
                    vis_bevs_right = change_occupancy_to_bev(semantics_right)
                    vis_bevs_left = change_occupancy_to_bev(semantics_left)
                    vis_bevs_straight = change_occupancy_to_bev(semantics_straight)
                    vis_bevs_rights.append(vis_bevs_right)
                    vis_bevs_lefts.append(vis_bevs_left)
                    vis_bevs_straights.append(vis_bevs_straight)
                save_path = os.path.join(save_root, 'forecast_bev_{}'.format(i))
                show_high_level_forecast_bev(save_path, vis_bevs_rights, vis_bevs_lefts, vis_bevs_straights)




if __name__ == '__main__':
    main()
