_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# nuscenes val scene=150, recommend use 6 gpus, 5 batchsize
# 6014it [00:00, 847798.35it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4519/4519 [00:53<00:00, 84.14it/s]
# evaluating time 0s ----------------------
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> others - IoU = 7.13
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> barrier - IoU = 31.18
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> bicycle - IoU = 18.74
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> bus - IoU = 32.55
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> car - IoU = 36.03
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> construction_vehicle - IoU = 17.5
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> motorcycle - IoU = 21.63
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> pedestrian - IoU = 21.47
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> traffic_cone - IoU = 20.58
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> trailer - IoU = 15.21
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> truck - IoU = 29.02
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> driveable_surface - IoU = 45.49
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> other_flat - IoU = 29.85
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> sidewalk - IoU = 30.74
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> terrain - IoU = 26.8
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> manmade - IoU = 19.53
# 2025-06-26 18:23:51,488 - mmdet3d - INFO - ===> vegetation - IoU = 23.8
# 2025-06-26 18:23:51,489 - mmdet3d - INFO - ===> empty - IoU = 94.9
# 2025-06-26 18:23:51,489 - mmdet3d - INFO - ===> mIoU of 4519 samples: 25.13
# 2025-06-26 18:23:51,489 - mmdet3d - INFO - ===> empty - IoU = 94.9
# 2025-06-26 18:23:51,489 - mmdet3d - INFO - ===> non-empty - IoU = 32.66
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4519/4519 [02:39<00:00, 28.28it/s]
# 2025-06-26 18:26:31,296 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> others - IoU = 6.56
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> barrier - IoU = 28.13
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> bicycle - IoU = 13.65
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> bus - IoU = 27.59
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> car - IoU = 28.02
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> construction_vehicle - IoU = 16.09
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> motorcycle - IoU = 14.36
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> pedestrian - IoU = 13.96
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> traffic_cone - IoU = 16.74
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> trailer - IoU = 13.72
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> truck - IoU = 24.71
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> driveable_surface - IoU = 43.63
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> other_flat - IoU = 27.71
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> sidewalk - IoU = 28.42
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> terrain - IoU = 24.76
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> manmade - IoU = 17.97
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> vegetation - IoU = 22.3
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> empty - IoU = 94.89
# 2025-06-26 18:26:31,297 - mmdet3d - INFO - ===> mIoU of 4519 samples: 21.67
# evaluating time 1.0s ----------------------
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> empty - IoU = 94.89
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> non-empty - IoU = 30.55
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> others - IoU = 6.05
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> barrier - IoU = 25.1
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> bicycle - IoU = 10.52
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> bus - IoU = 21.14
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> car - IoU = 21.11
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> construction_vehicle - IoU = 14.72
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> motorcycle - IoU = 11.03
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> pedestrian - IoU = 8.77
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> traffic_cone - IoU = 13.51
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> trailer - IoU = 11.41
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> truck - IoU = 20.19
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> driveable_surface - IoU = 42.12
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> other_flat - IoU = 26.18
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> sidewalk - IoU = 26.85
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> terrain - IoU = 23.32
# 2025-06-26 18:26:31,298 - mmdet3d - INFO - ===> manmade - IoU = 16.52
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> vegetation - IoU = 20.76
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> empty - IoU = 94.88
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> mIoU of 4519 samples: 18.78
# evaluating time 2.0s ----------------------
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> empty - IoU = 94.88
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> non-empty - IoU = 28.76
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> others - IoU = 5.54
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> barrier - IoU = 22.24
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> bicycle - IoU = 8.35
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> bus - IoU = 15.78
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> car - IoU = 16.94
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> construction_vehicle - IoU = 13.31
# 2025-06-26 18:26:31,299 - mmdet3d - INFO - ===> motorcycle - IoU = 8.89
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> pedestrian - IoU = 5.66
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> traffic_cone - IoU = 10.67
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> trailer - IoU = 9.31
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> truck - IoU = 16.94
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> driveable_surface - IoU = 40.41
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> other_flat - IoU = 24.46
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> sidewalk - IoU = 25.24
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> terrain - IoU = 21.95
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> manmade - IoU = 15.1
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> vegetation - IoU = 19.17
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> empty - IoU = 94.91
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> mIoU of 4519 samples: 16.47
# evaluating time 3.0s ----------------------
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> empty - IoU = 94.91
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> non-empty - IoU = 26.99
# 2025-06-26 18:26:31,300 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> others - IoU = 6.56
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> barrier - IoU = 28.13
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> bicycle - IoU = 13.65
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> bus - IoU = 27.59
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> car - IoU = 28.02
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> construction_vehicle - IoU = 16.09
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> motorcycle - IoU = 14.36
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> pedestrian - IoU = 13.96
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> traffic_cone - IoU = 16.74
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> trailer - IoU = 13.72
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> truck - IoU = 24.71
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> driveable_surface - IoU = 43.63
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> other_flat - IoU = 27.71
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> sidewalk - IoU = 28.42
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> terrain - IoU = 24.76
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> manmade - IoU = 17.97
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> vegetation - IoU = 22.3
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> empty - IoU = 94.89
# 2025-06-26 18:26:31,301 - mmdet3d - INFO - ===> mIoU of 4519 samples: 21.67
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> empty - IoU = 94.89
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> non-empty - IoU = 30.55
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> others - IoU = 6.05
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> barrier - IoU = 25.1
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> bicycle - IoU = 10.52
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> bus - IoU = 21.14
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> car - IoU = 21.11
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> construction_vehicle - IoU = 14.72
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> motorcycle - IoU = 11.03
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> pedestrian - IoU = 8.77
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> traffic_cone - IoU = 13.51
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> trailer - IoU = 11.41
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> truck - IoU = 20.19
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> driveable_surface - IoU = 42.12
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> other_flat - IoU = 26.18
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> sidewalk - IoU = 26.85
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> terrain - IoU = 23.32
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> manmade - IoU = 16.52
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> vegetation - IoU = 20.76
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> empty - IoU = 94.88
# 2025-06-26 18:26:31,302 - mmdet3d - INFO - ===> mIoU of 4519 samples: 18.78
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> empty - IoU = 94.88
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> non-empty - IoU = 28.76
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> per class IoU of 4519 samples:
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> others - IoU = 5.54
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> barrier - IoU = 22.24
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> bicycle - IoU = 8.35
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> bus - IoU = 15.78
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> car - IoU = 16.94
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> construction_vehicle - IoU = 13.31
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> motorcycle - IoU = 8.89
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> pedestrian - IoU = 5.66
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> traffic_cone - IoU = 10.67
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> trailer - IoU = 9.31
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> truck - IoU = 16.94
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> driveable_surface - IoU = 40.41
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> other_flat - IoU = 24.46
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> sidewalk - IoU = 25.24
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> terrain - IoU = 21.95
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> manmade - IoU = 15.1
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> vegetation - IoU = 19.17
# 2025-06-26 18:26:31,303 - mmdet3d - INFO - ===> empty - IoU = 94.91
# 2025-06-26 18:26:31,304 - mmdet3d - INFO - ===> mIoU of 4519 samples: 16.47
# 2025-06-26 18:26:31,304 - mmdet3d - INFO - ===> empty - IoU = 94.91
# 2025-06-26 18:26:31,304 - mmdet3d - INFO - ===> non-empty - IoU = 26.99
# 2025-06-26 18:26:31,304 - mmdet3d - INFO - Evaluation Results:
# 2025-06-26 18:26:31,304 - mmdet3d - INFO - +---------+-------+-------+
# | Time    | mIoU  | IoU   |
# +---------+-------+-------+
# | 1.0s    | 21.67 | 30.55 |
# | 2.0s    | 18.78 | 28.76 |
# | 3.0s    | 16.47 | 26.99 |
# | Average | 18.97 | 28.77 |
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
    dict(type='LoadStreamLatentToken', data_path='data/nuscenes/save_dir_stc/token_4f'),
    dict(type='Collect3D', keys=['latent'])
]

test_pipeline = [
    dict(type='LoadStreamLatentToken', data_path='data/nuscenes/save_dir_stc/token_4f'),
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
    ann_file=data_root + 'world-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'world-nuscenes_infos_train.pkl',
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