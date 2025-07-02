_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# 2025-06-27 10:37:12,742 - mmdet3d - INFO - ===> per class IoU of 6014 samples:
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> others - IoU = 7.25
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> barrier - IoU = 30.8
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> bicycle - IoU = 18.17
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> bus - IoU = 31.98
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> car - IoU = 35.8
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> construction_vehicle - IoU = 17.46
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> motorcycle - IoU = 21.18
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> pedestrian - IoU = 21.07
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> traffic_cone - IoU = 20.75
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> trailer - IoU = 14.88
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> truck - IoU = 28.46
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> driveable_surface - IoU = 44.93
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> other_flat - IoU = 29.6
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> sidewalk - IoU = 30.13
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> terrain - IoU = 26.31
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> manmade - IoU = 19.39
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> vegetation - IoU = 23.39
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> empty - IoU = 95.03
# 2025-06-27 10:37:12,743 - mmdet3d - INFO - ===> mIoU of 6014 samples: 24.8
# 2025-06-27 10:37:12,744 - mmdet3d - INFO - ===> empty - IoU = 95.03
# 2025-06-27 10:37:12,744 - mmdet3d - INFO - ===> non-empty - IoU = 32.24
# {'semantics_miou': 24.8, 'binary_iou': 32.24}

# nuscenes val scene=150, recommend use 6 gpus, 5 batchsize
# Dataset Config
dataset_name = 'occ3d'
eval_metric = 'miou'

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
}
# grid config corresponding to the voxel size, x: 80/1.6=50, y: 80/1.6=50, z: 6.4/1.6=4

train_load_future_frame_number = 0
test_load_future_frame_number = 0
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
total_epoch = 24
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu)*4.554)      # total samples: 28130

# Model Config

# others params
num_classes = len(occ_class_names)
z_height = 16

base_channel = 64
class_embeds_dim = 16
n_e_ = 512
frame_number = 4
model = dict(
    type='IISceneTokenizer',
    empty_idx=occ_class_names.index('free'),
    class_weights=class_weights,
    num_classes=num_classes,
    class_embeds_dim=class_embeds_dim,
    embed_loss_weight=1.0,
    frame_number=frame_number,
    vq_channel=base_channel * 2,
    grid_config=grid_config,
    encoder=dict(
        type='Encoder2D',
        ch = base_channel,
        out_ch = base_channel,
        ch_mult = (1,2,4),
        num_res_blocks = (2, 2, 4),
        attn_resolutions = (50,),
        dropout = 0.0,
        resamp_with_conv = True,
        in_channels = z_height * class_embeds_dim,
        resolution = 200,
        z_channels = base_channel * 2,
        double_z = False,
    ),
    vq=dict(
        type='IntraInterVectorQuantizer',
        n_e = n_e_,
        e_dim = base_channel * 2,
        beta = 1.,
        z_channels = base_channel * 2,
        recover_time=frame_number,
        use_voxel=False
    ),
    decoder=dict(
        type='Decoder2D',
        ch = base_channel,
        out_ch = z_height * class_embeds_dim,
        ch_mult = (1,2,4),
        num_res_blocks = (2, 2, 4),
        attn_resolutions = (50,),
        dropout = 0.0,
        resamp_with_conv = True,
        in_channels = z_height * class_embeds_dim,
        resolution = 200,
        z_channels = base_channel * 2,
        give_pre_end = False
    ),
    focal_loss=dict(
        type='CustomFocalLoss',
        loss_weight=10.0,
    )
)

# Data
dataset_type = 'NuScenesWorldDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True, dataset_type='stcocc'),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='Collect3D', keys=['voxel_semantics'])
]

test_pipeline = [
    dict(type='LoadStreamOcc3D', to_long=True, dataset_type='stcocc'),
    dict(type='BEVAugStream', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='Collect3D', keys=['voxel_semantics'])
]

share_data_config = dict(
    type=dataset_type,
    classes=occ_class_names,
    use_sequence_group_flag=True,
    # Eval Config
    dataset_name=dataset_name,
    eval_metric=eval_metric,
)

test_data_config = dict(
    pipeline=test_pipeline,
    load_future_frame_number=test_load_future_frame_number,
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
        # Video Sequence
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
    ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 5e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2),)

step_epoch = 20
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
    )
]

revise_keys = None
load_from=None