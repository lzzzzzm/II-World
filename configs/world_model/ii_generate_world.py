_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# nuscenes val scene=150, recommend use 6 gpus, 5 batchsize
# evaluating time 0s ----------------------
# 2025-05-16 10:31:11,165 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-05-16 10:31:11,169 - mmdet3d - INFO - ===> others - IoU = 73.72
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> barrier - IoU = 93.73
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> bicycle - IoU = 95.29
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> bus - IoU = 82.47
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> car - IoU = 81.91
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> construction_vehicle - IoU = 68.93
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> motorcycle - IoU = 95.67
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> pedestrian - IoU = 93.56
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> traffic_cone - IoU = 95.11
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> trailer - IoU = 75.15
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> truck - IoU = 79.39
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> driveable_surface - IoU = 85.91
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> other_flat - IoU = 94.35
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> sidewalk - IoU = 78.84
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> terrain - IoU = 76.83
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> manmade - IoU = 59.81
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> vegetation - IoU = 50.04
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> empty - IoU = 97.99
# 2025-05-16 10:31:11,170 - mmdet3d - INFO - ===> mIoU of 4519 samples: 81.22
# 2025-05-16 10:31:11,175 - mmdet3d - INFO - ===> empty - IoU = 97.99
# 2025-05-16 10:31:11,175 - mmdet3d - INFO - ===> non-empty - IoU = 68.3
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4519/4519 [04:10<00:00, 18.05it/s]
# evaluating time 1.0s ----------------------
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> others - IoU = 49.05
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> barrier - IoU = 58.94
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> bicycle - IoU = 36.02
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> bus - IoU = 49.09
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> car - IoU = 46.69
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> construction_vehicle - IoU = 45.63
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> motorcycle - IoU = 34.88
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> pedestrian - IoU = 28.51
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> traffic_cone - IoU = 40.51
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> trailer - IoU = 47.96
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> truck - IoU = 51.68
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> driveable_surface - IoU = 66.36
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> other_flat - IoU = 55.29
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> sidewalk - IoU = 54.9
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> terrain - IoU = 53.91
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> manmade - IoU = 46.84
# 2025-05-16 10:35:21,521 - mmdet3d - INFO - ===> vegetation - IoU = 43.3
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> empty - IoU = 96.94
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> mIoU of 4519 samples: 47.62
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> empty - IoU = 96.94
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> non-empty - IoU = 54.29
# evaluating time 2.0s ----------------------
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> others - IoU = 41.98
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> barrier - IoU = 49.61
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> bicycle - IoU = 25.79
# 2025-05-16 10:35:21,522 - mmdet3d - INFO - ===> bus - IoU = 33.76
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> car - IoU = 33.52
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> construction_vehicle - IoU = 37.31
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> motorcycle - IoU = 22.93
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> pedestrian - IoU = 15.26
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> traffic_cone - IoU = 30.52
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> trailer - IoU = 37.47
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> truck - IoU = 39.1
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> driveable_surface - IoU = 61.39
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> other_flat - IoU = 48.77
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> sidewalk - IoU = 49.13
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> terrain - IoU = 48.17
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> manmade - IoU = 41.83
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> vegetation - IoU = 39.39
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> empty - IoU = 96.6
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> mIoU of 4519 samples: 38.58
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> empty - IoU = 96.6
# 2025-05-16 10:35:21,523 - mmdet3d - INFO - ===> non-empty - IoU = 49.43
# evaluating time 3.0s ----------------------
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> others - IoU = 37.2
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> barrier - IoU = 42.99
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> bicycle - IoU = 19.4
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> bus - IoU = 24.55
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> car - IoU = 26.73
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> construction_vehicle - IoU = 30.97
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> motorcycle - IoU = 17.79
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> pedestrian - IoU = 9.36
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> traffic_cone - IoU = 24.15
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> trailer - IoU = 31.34
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> truck - IoU = 31.56
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> driveable_surface - IoU = 57.67
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> other_flat - IoU = 44.42
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> sidewalk - IoU = 44.93
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> terrain - IoU = 43.88
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> manmade - IoU = 37.82
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> vegetation - IoU = 35.94
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> empty - IoU = 96.39
# 2025-05-16 10:35:21,524 - mmdet3d - INFO - ===> mIoU of 4519 samples: 32.98
# 2025-05-16 10:35:21,525 - mmdet3d - INFO - ===> empty - IoU = 96.39
# 2025-05-16 10:35:21,525 - mmdet3d - INFO - ===> non-empty - IoU = 45.69
# 2025-05-16 10:35:21,529 - mmdet3d - INFO - Evaluation Results:
# 2025-05-16 10:35:21,529 - mmdet3d - INFO - +---------+-------+-------+
# | Time    | mIoU  | IoU   |
# +---------+-------+-------+
# | 1.0s    | 47.62 | 54.29 |
# | 2.0s    | 38.58 | 49.43 |
# | 3.0s    | 32.98 | 45.69 |
# | Average | 39.73 | 49.80 |
# +---------+-------+-------+
# Dataset Config
dataset_name = 'occ3d'
eval_metric = 'forecasting_miou'

class_weights = [0.0727, 0.0692, 0.0838, 0.0681, 0.0601, 0.0741, 0.0823, 0.0688, 0.0773, 0.0681, 0.0641, 0.0527, 0.0655, 0.0563, 0.0558, 0.0541, 0.0538, 0.0468] # occ-3d

occ_class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle','motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation','free']   # occ3d

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

grid_config = {
    'x': [-40, 40, 1.6],
    'y': [-40, 40, 1.6],
    'z': [-1, 5.4, 1.6],
    'depth': [1.0, 45.0, 0.5],
}

# In nuscenes data label is marked in 0.5s interval, so we can load future frame 6 to predict 3s future
train_load_future_frame_number = 6      # 0.5s interval 1 frame
train_load_previous_frame_number = 0    # 0.5s interval 1 frame
test_load_future_frame_number = 6       # 0.5s interval 1 frame
test_load_previous_frame_number = 4     # 0.5s interval 1 frame

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# Running Config
num_gpus = 8
samples_per_gpu = 8
workers_per_gpu = 4
total_epoch = 48
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu)*4.554)      # total samples: 28130

# Model Config

# others params
num_classes = len(occ_class_names)
base_channel = 64
z_height = 16
class_embeds_dim = 16
n_e_ = 512
vq_frame_number = 4
previous_frame = 4
future_frame = 6
embed_dims = base_channel * 2
_ffn_dim_ = embed_dims * 2
pos_dim = embed_dims // 2

row_num_embed = 50  # latent_height
col_num_embed = 50  # latent_width

memory_frame_number = 5 # 4 history frames + 1 current frame
task_mode = 'generate'
model = dict(
    type='II_World',
    previous_frame_exist=True if train_load_previous_frame_number > 0 else False,
    previous_frame=previous_frame,
    train_future_frame=train_load_future_frame_number,
    test_future_frame=test_load_future_frame_number,
    test_previous_frame=test_load_previous_frame_number,
    memory_frame_number=memory_frame_number,
    task_mode=task_mode,
    test_mode=False,
    feature_similarity_loss=dict(
        type='FeatSimLoss',
        loss_weight=1.0,
    ),
    trajs_loss=dict(
        type='TrajLoss',
        loss_weight=0.01,
    ),
    rotation_loss=dict(
        type='RotationLoss',
        loss_weight=1.0,
    ),
    pose_encoder=dict(
        type='PoseEncoder',
        history_frame_number=memory_frame_number,
    ),
    transformer=dict(
        type='II_Former',
        embed_dims=embed_dims,
        output_dims=embed_dims,
        use_gt_traj=True,
        use_transformation=True,
        history_frame_number=memory_frame_number,
        task_mode=task_mode,
        low_encoder=dict(
            type='II_FormerEncoder',
            num_layers=3,
            return_intermediate=True,
            transformerlayers=dict(
                type='II_FormerEncoderLayer',
                use_plan=True,
                attn_cfgs=[
                    dict(
                        type='SelfAttention',
                        embed_dims=embed_dims,
                        dropout=0.0,
                        num_levels=1,
                    ),
                    dict(
                        type='CrossPlanAttention',
                        embed_dims=embed_dims,
                        dropout=0.0,
                        num_levels=1,
                    )
                ],
                conv_cfgs=dict(
                    embed_dims=embed_dims,
                    stride=2,
                ),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'conv')
            )
        ),
        high_encoder=dict(
            type='II_FormerEncoder',
            num_layers=3,
            transformerlayers=dict(
                type='II_FormerEncoderLayer',
                use_plan=False,
                attn_cfgs=[
                    dict(
                        type='SelfAttention',
                        embed_dims=embed_dims,
                        dropout=0.0,
                        num_levels=1,
                    ),
                    dict(
                        type='TemporalFusion',
                        embed_dims=embed_dims,
                        hisotry_number=memory_frame_number,
                        dropout=0.0,
                        num_levels=1,
                    )
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=embed_dims,
                    feedforward_channels=_ffn_dim_,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=('self_attn', 'norm', 'temporal_fusion', 'norm', 'ffn', 'norm')
            )
        ),
        positional_encoding=dict(
            type='PositionalEncoding',
            num_feats=pos_dim,
            row_num_embed=row_num_embed,
            col_num_embed=col_num_embed,
        )
    ),
    vqvae=dict(
        type='IISceneTokenizer',
        empty_idx=occ_class_names.index('free'),
        class_weights=class_weights,
        num_classes=num_classes,
        class_embeds_dim=class_embeds_dim,
        embed_loss_weight=1.0,
        frame_number=4,
        vq_channel=base_channel * 2,
        grid_config=grid_config,
        encoder=dict(
            type='Encoder2D',
            ch=base_channel,
            out_ch=base_channel,
            ch_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 4),
            attn_resolutions=(50,),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=z_height * class_embeds_dim,
            resolution=200,
            z_channels=base_channel * 2,
            double_z=False,
        ),
        vq=dict(
            type='IntraInterVectorQuantizer',
            n_e=n_e_,
            e_dim=base_channel * 2,
            beta=1.,
            z_channels=base_channel * 2,
            recover_time=4,
            use_voxel=False
        ),
        decoder=dict(
            type='Decoder2D',
            ch=base_channel,
            out_ch=z_height * class_embeds_dim,
            ch_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 4),
            attn_resolutions=(50,),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=z_height * class_embeds_dim,
            resolution=200,
            z_channels=base_channel * 2,
            give_pre_end=False
        ),
        focal_loss=dict(
            type='CustomFocalLoss',
            loss_weight=10.0,
        )
    )
)

# Data
dataset_type = 'NuScenesWorldDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadStreamLatentToken', data_path='data/nuscenes/save_dir/token_4f'),
    dict(type='Collect3D', keys=['latent'])
]

test_pipeline = [
    dict(type='LoadStreamLatentToken', data_path='data/nuscenes/save_dir/token_4f'),
    dict(type='LoadStreamOcc3D'),
    dict(type='Collect3D', keys=['voxel_semantics', 'latent'])
]

share_data_config = dict(
    type=dataset_type,
    classes=occ_class_names,
    use_sequence_group_flag=True,
    # Eval Config
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    load_previous_data=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    load_future_frame_number=test_load_future_frame_number,
    load_previous_frame_number=test_load_previous_frame_number,
    ann_file=data_root + 'world-lidar-v2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'world-lidar-v2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=occ_class_names,
        test_mode=False,
        load_future_frame_number=train_load_future_frame_number,
        load_previous_frame_number=train_load_previous_frame_number,
        # Video Sequence
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        # Set BEV Augmentation for the same sequence
        # bda_aug_conf=bda_aug_conf,
    ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 1e-3
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2),)

step_epoch = 36
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch*step_epoch,])

checkpoint_epoch_interval = 1
runner = dict(type='IterBasedRunner', max_iters=total_epoch * num_iters_per_epoch)
checkpoint_config = dict(interval=checkpoint_epoch_interval * num_iters_per_epoch)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2*num_iters_per_epoch,
    ),
    dict(
        type='ScheduledSampling',
        total_iter=total_epoch * num_iters_per_epoch,
        loss_iter=None,
        # trans_iter=num_iters_per_epoch*step_epoch
    )
]

revise_keys = None
# load_from='ckpts/mxworld_pose.pth'