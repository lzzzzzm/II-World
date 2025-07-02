_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# Due to waymo validation 202, we use 2 gpus to evaluate the model.
# Due to large waymo dataset, when utilize sample hz = 10, set load_interval=1, utilize the eval_metric = 'miou', and set the eval_time for 1, 3, 5 separately.
# Sample hz=10
# 1s
# 2025-06-30 12:14:32,789 - mmdet3d - INFO - ===> per class IoU of 39962 samples:
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> others - IoU = 54.33
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> bicycle - IoU = 42.18
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> car - IoU = 47.73
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> motorcycle - IoU = 28.85
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> pedestrian - IoU = 28.08
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> traffic_cone - IoU = 42.51
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> driveable_surface - IoU = 77.45
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> sidewalk - IoU = 68.92
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> manmade - IoU = 50.77
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> vegetation - IoU = 52.69
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> empty - IoU = 96.82
# 2025-06-30 12:14:32,790 - mmdet3d - INFO - ===> mIoU of 39962 samples: 49.35
# 2025-06-30 12:14:32,791 - mmdet3d - INFO - ===> empty - IoU = 96.82
# 2025-06-30 12:14:32,791 - mmdet3d - INFO - ===> non-empty - IoU = 64.58
# {'semantics_miou': 49.35, 'binary_iou': 64.58}
# 2s
# 2025-06-30 13:32:33,135 - mmdet3d - INFO - ===> per class IoU of 39962 samples:
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> others - IoU = 50.51
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> bicycle - IoU = 31.13
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> car - IoU = 41.29
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> motorcycle - IoU = 23.04
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> pedestrian - IoU = 14.5
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> traffic_cone - IoU = 36.77
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> driveable_surface - IoU = 72.79
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> sidewalk - IoU = 62.83
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> manmade - IoU = 47.39
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> vegetation - IoU = 50.02
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> empty - IoU = 96.43
# 2025-06-30 13:32:33,136 - mmdet3d - INFO - ===> mIoU of 39962 samples: 43.03
# 2025-06-30 13:32:33,137 - mmdet3d - INFO - ===> empty - IoU = 96.43
# 2025-06-30 13:32:33,137 - mmdet3d - INFO - ===> non-empty - IoU = 60.71
# {'semantics_miou': 43.03, 'binary_iou': 60.71}
# 3s
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> per class IoU of 39962 samples:
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> others - IoU = 47.2
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> bicycle - IoU = 25.64
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> car - IoU = 37.97
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> motorcycle - IoU = 17.57
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> pedestrian - IoU = 10.18
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> traffic_cone - IoU = 32.04
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> driveable_surface - IoU = 69.24
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> sidewalk - IoU = 58.08
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-06-30 16:32:00,140 - mmdet3d - INFO - ===> manmade - IoU = 44.47
# 2025-06-30 16:32:00,141 - mmdet3d - INFO - ===> vegetation - IoU = 47.51
# 2025-06-30 16:32:00,141 - mmdet3d - INFO - ===> empty - IoU = 96.12
# 2025-06-30 16:32:00,141 - mmdet3d - INFO - ===> mIoU of 39962 samples: 38.99
# 2025-06-30 16:32:00,141 - mmdet3d - INFO - ===> empty - IoU = 96.12
# 2025-06-30 16:32:00,141 - mmdet3d - INFO - ===> non-empty - IoU = 57.63
# {'semantics_miou': 38.99, 'binary_iou': 57.63}
# | Time    | mIoU  | IoU   |
# +---------+-------+-------+
# | 1.0s    | 49.35 | 64.58 |
# | 2.0s    | 43.03 | 60.71 |
# | 3.0s    | 38.99 | 57.63 |
# | Average | 43.73 | 60.97 |
# +---------+-------+-------+

# Sample Hz = 2
# 7994it [00:00, 1885679.44it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7994/7994 [06:13<00:00, 21.38it/s]
# 2025-07-02 11:46:56,889 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> others - IoU = 50.16
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> bicycle - IoU = 31.64
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> car - IoU = 42.53
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> motorcycle - IoU = 26.45
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> pedestrian - IoU = 21.66
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> traffic_cone - IoU = 35.26
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> driveable_surface - IoU = 69.27
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> sidewalk - IoU = 58.01
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> manmade - IoU = 45.9
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> vegetation - IoU = 47.51
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> empty - IoU = 96.21
# 2025-07-02 11:46:56,890 - mmdet3d - INFO - ===> mIoU of 7994 samples: 42.84
# evaluating time 1s ----------------------
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> empty - IoU = 96.21
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> non-empty - IoU = 58.29
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> others - IoU = 45.03
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> bicycle - IoU = 24.73
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> car - IoU = 36.0
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> motorcycle - IoU = 20.07
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> pedestrian - IoU = 12.28
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> traffic_cone - IoU = 25.17
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> driveable_surface - IoU = 62.16
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> sidewalk - IoU = 49.01
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> manmade - IoU = 40.17
# 2025-07-02 11:46:56,891 - mmdet3d - INFO - ===> vegetation - IoU = 41.18
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> empty - IoU = 95.61
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> mIoU of 7994 samples: 35.58
# evaluating time 2s ----------------------
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> empty - IoU = 95.61
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> non-empty - IoU = 51.84
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> others - IoU = 40.94
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> bicycle - IoU = 20.45
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> car - IoU = 32.24
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> motorcycle - IoU = 14.77
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> pedestrian - IoU = 8.42
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> traffic_cone - IoU = 18.95
# 2025-07-02 11:46:56,892 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> driveable_surface - IoU = 56.86
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> sidewalk - IoU = 42.72
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> manmade - IoU = 35.89
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> vegetation - IoU = 35.96
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> empty - IoU = 95.19
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> mIoU of 7994 samples: 30.72
# evaluating time 3s ----------------------
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> empty - IoU = 95.19
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> non-empty - IoU = 46.94
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> others - IoU = 50.16
# 2025-07-02 11:46:56,893 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> bicycle - IoU = 31.64
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> car - IoU = 42.53
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> motorcycle - IoU = 26.45
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> pedestrian - IoU = 21.66
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> traffic_cone - IoU = 35.26
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> driveable_surface - IoU = 69.27
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> sidewalk - IoU = 58.01
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> manmade - IoU = 45.9
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> vegetation - IoU = 47.51
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> empty - IoU = 96.21
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> mIoU of 7994 samples: 42.84
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> empty - IoU = 96.21
# 2025-07-02 11:46:56,894 - mmdet3d - INFO - ===> non-empty - IoU = 58.29
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> others - IoU = 45.03
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> bicycle - IoU = 24.73
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> car - IoU = 36.0
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> motorcycle - IoU = 20.07
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> pedestrian - IoU = 12.28
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> traffic_cone - IoU = 25.17
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> driveable_surface - IoU = 62.16
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> sidewalk - IoU = 49.01
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> manmade - IoU = 40.17
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> vegetation - IoU = 41.18
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> empty - IoU = 95.61
# 2025-07-02 11:46:56,895 - mmdet3d - INFO - ===> mIoU of 7994 samples: 35.58
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> empty - IoU = 95.61
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> non-empty - IoU = 51.84
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> others - IoU = 40.94
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> bicycle - IoU = 20.45
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> car - IoU = 32.24
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> motorcycle - IoU = 14.77
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> pedestrian - IoU = 8.42
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> traffic_cone - IoU = 18.95
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> driveable_surface - IoU = 56.86
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> sidewalk - IoU = 42.72
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> manmade - IoU = 35.89
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> vegetation - IoU = 35.96
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> empty - IoU = 95.19
# 2025-07-02 11:46:56,896 - mmdet3d - INFO - ===> mIoU of 7994 samples: 30.72
# 2025-07-02 11:46:56,897 - mmdet3d - INFO - ===> empty - IoU = 95.19
# 2025-07-02 11:46:56,897 - mmdet3d - INFO - ===> non-empty - IoU = 46.94
# 2025-07-02 11:46:56,897 - mmdet3d - INFO - Evaluation Results:
# 2025-07-02 11:46:56,897 - mmdet3d - INFO - +---------+-------+-------+
# | Time    | mIoU  | IoU   |
# +---------+-------+-------+
# | 1s      | 42.84 | 58.29 |
# | 2s      | 35.58 | 51.84 |
# | 3s      | 30.72 | 46.94 |
# | Average | 36.38 | 52.36 |
# +---------+-------+-------+


# --------------- Copy Past ----------------------#
# Sample hz=10
# 1s
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> per class IoU of 39962 samples:
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> others - IoU = 37.44
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> bicycle - IoU = 31.85
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> car - IoU = 41.73
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> motorcycle - IoU = 15.72
# 2025-07-01 16:40:51,161 - mmdet3d - INFO - ===> pedestrian - IoU = 33.82
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> traffic_cone - IoU = 10.8
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> driveable_surface - IoU = 69.49
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> sidewalk - IoU = 54.73
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> manmade - IoU = 27.69
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> vegetation - IoU = 27.34
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> empty - IoU = 94.52
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> mIoU of 39962 samples: 35.06
# 2025-07-01 16:40:51,162 - mmdet3d - INFO - ===> empty - IoU = 94.52
# 2025-07-01 16:40:51,163 - mmdet3d - INFO - ===> non-empty - IoU = 46.46
# {'semantics_miou': 35.06, 'binary_iou': 46.46}

# 2s
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> per class IoU of 39962 samples:
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> others - IoU = 30.87
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> bicycle - IoU = 20.21
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> car - IoU = 30.02
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> motorcycle - IoU = 11.2
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> pedestrian - IoU = 21.45
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> traffic_cone - IoU = 9.19
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> driveable_surface - IoU = 60.45
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> sidewalk - IoU = 43.52
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> manmade - IoU = 22.28
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> vegetation - IoU = 19.18
# 2025-07-02 00:40:44,846 - mmdet3d - INFO - ===> empty - IoU = 93.41
# 2025-07-02 00:40:44,847 - mmdet3d - INFO - ===> mIoU of 39962 samples: 26.83
# 2025-07-02 00:40:44,847 - mmdet3d - INFO - ===> empty - IoU = 93.41
# 2025-07-02 00:40:44,847 - mmdet3d - INFO - ===> non-empty - IoU = 38.69
# {'semantics_miou': 26.83, 'binary_iou': 38.69}

# 3s
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> per class IoU of 39962 samples:
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> others - IoU = 27.54
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> bicycle - IoU = 15.2
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> car - IoU = 25.01
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> motorcycle - IoU = 9.93
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> pedestrian - IoU = 15.47
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> traffic_cone - IoU = 9.05
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-01 13:34:53,346 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> driveable_surface - IoU = 55.51
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> sidewalk - IoU = 37.93
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> manmade - IoU = 19.75
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> vegetation - IoU = 15.97
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> empty - IoU = 92.86
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> mIoU of 39962 samples: 23.14
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> empty - IoU = 92.86
# 2025-07-01 13:34:53,347 - mmdet3d - INFO - ===> non-empty - IoU = 35.13
# {'semantics_miou': 23.14, 'binary_iou': 35.13}
# | Time    | mIoU  | IoU   |
# +---------+-------+-------+
# | 1s      | 35.06 | 46.46 |
# | 2s      | 26.83 | 38.69 |
# | 3s      | 23.14 | 35.13 |
# | Average | 28.34 | 40.09 |
# +---------+-------+-------+

# Sample hz=2
# 7994it [00:00, 1875868.09it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7994/7994 [07:16<00:00, 18.30it/s]
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> others - IoU = 23.93
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> bicycle - IoU = 12.52
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> car - IoU = 21.09
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> motorcycle - IoU = 9.38
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> pedestrian - IoU = 10.85
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> traffic_cone - IoU = 9.17
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> driveable_surface - IoU = 49.3
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> sidewalk - IoU = 31.9
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> manmade - IoU = 17.63
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> vegetation - IoU = 13.66
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> empty - IoU = 92.28
# 2025-07-02 12:14:11,599 - mmdet3d - INFO - ===> mIoU of 7994 samples: 19.94
# evaluating time 1s ----------------------
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> empty - IoU = 92.28
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> non-empty - IoU = 31.53
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> others - IoU = 19.34
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> bicycle - IoU = 9.75
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> car - IoU = 17.04
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> motorcycle - IoU = 8.46
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> pedestrian - IoU = 8.49
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> traffic_cone - IoU = 8.48
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> driveable_surface - IoU = 42.48
# 2025-07-02 12:14:11,600 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> sidewalk - IoU = 25.77
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> manmade - IoU = 14.81
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> vegetation - IoU = 11.33
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> empty - IoU = 91.58
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> mIoU of 7994 samples: 16.6
# evaluating time 2s ----------------------
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> empty - IoU = 91.58
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> non-empty - IoU = 27.38
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> others - IoU = 16.88
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> bicycle - IoU = 8.24
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 12:14:11,601 - mmdet3d - INFO - ===> car - IoU = 15.29
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> motorcycle - IoU = 7.76
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> pedestrian - IoU = 7.41
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> traffic_cone - IoU = 7.85
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> driveable_surface - IoU = 39.41
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> sidewalk - IoU = 23.1
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> manmade - IoU = 13.35
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> vegetation - IoU = 10.35
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> empty - IoU = 91.26
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> mIoU of 7994 samples: 14.96
# evaluating time 3s ----------------------
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> empty - IoU = 91.26
# 2025-07-02 12:14:11,602 - mmdet3d - INFO - ===> non-empty - IoU = 25.48
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> others - IoU = 23.93
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> bicycle - IoU = 12.52
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> car - IoU = 21.09
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> motorcycle - IoU = 9.38
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> pedestrian - IoU = 10.85
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> traffic_cone - IoU = 9.17
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> driveable_surface - IoU = 49.3
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> sidewalk - IoU = 31.9
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> manmade - IoU = 17.63
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> vegetation - IoU = 13.66
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> empty - IoU = 92.28
# 2025-07-02 12:14:11,603 - mmdet3d - INFO - ===> mIoU of 7994 samples: 19.94
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> empty - IoU = 92.28
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> non-empty - IoU = 31.53
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> others - IoU = 19.34
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> bicycle - IoU = 9.75
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> car - IoU = 17.04
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> motorcycle - IoU = 8.46
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> pedestrian - IoU = 8.49
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> traffic_cone - IoU = 8.48
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> driveable_surface - IoU = 42.48
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> sidewalk - IoU = 25.77
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> manmade - IoU = 14.81
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> vegetation - IoU = 11.33
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> empty - IoU = 91.58
# 2025-07-02 12:14:11,604 - mmdet3d - INFO - ===> mIoU of 7994 samples: 16.6
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> empty - IoU = 91.58
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> non-empty - IoU = 27.38
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> per class IoU of 7994 samples:
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> others - IoU = 16.88
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> barrier - IoU = nan
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> bicycle - IoU = 8.24
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> bus - IoU = nan
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> car - IoU = 15.29
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> construction_vehicle - IoU = nan
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> motorcycle - IoU = 7.76
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> pedestrian - IoU = 7.41
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> traffic_cone - IoU = 7.85
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> trailer - IoU = nan
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> truck - IoU = nan
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> driveable_surface - IoU = 39.41
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> other_flat - IoU = nan
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> sidewalk - IoU = 23.1
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> terrain - IoU = nan
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> manmade - IoU = 13.35
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> vegetation - IoU = 10.35
# 2025-07-02 12:14:11,605 - mmdet3d - INFO - ===> empty - IoU = 91.26
# 2025-07-02 12:14:11,606 - mmdet3d - INFO - ===> mIoU of 7994 samples: 14.96
# 2025-07-02 12:14:11,606 - mmdet3d - INFO - ===> empty - IoU = 91.26
# 2025-07-02 12:14:11,606 - mmdet3d - INFO - ===> non-empty - IoU = 25.48
# 2025-07-02 12:14:11,606 - mmdet3d - INFO - Evaluation Results:
# 2025-07-02 12:14:11,606 - mmdet3d - INFO - +---------+-------+-------+
# | Time    | mIoU  | IoU   |
# +---------+-------+-------+
# | 1s      | 19.94 | 31.53 |
# | 2s      | 16.60 | 27.38 |
# | 3s      | 14.96 | 25.48 |
# | Average | 17.17 | 28.13 |
# +---------+-------+-------+

# Dataset Config
dataset_name = 'waymo-Occ3D'
eval_metric = 'forecasting_miou'
load_interval = 5

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
eval_time = 3
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
    dataset_type='waymo',
    eval_time=eval_time,
    eval_metric=eval_metric,
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
dataset_type = 'WaymoWorldDataset'
data_root = 'data/waymo/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadStreamLatentToken', data_path='data/waymo/save_dir/token_4f', dataset_type='waymo'),
    dict(type='Collect3D', keys=['latent'])
]

test_pipeline = [
    dict(type='LoadStreamLatentToken', data_path='data/waymo/save_dir/token_4f', dataset_type='waymo'),
    dict(type='LoadStreamOcc3D', dataset_type='waymo'),
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
    load_interval=load_interval,
)

test_data_config = dict(
    pipeline=test_pipeline,
    load_future_frame_number=test_load_future_frame_number,
    load_previous_frame_number=test_load_previous_frame_number,
    ann_file=data_root + 'waymo_infos_val.pkl',
    pose_file=data_root + 'cam_infos_vali.pkl',
    split='validation',
    data_root=data_root,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        pose_file=data_root + 'cam_infos_vali.pkl',
        split='validation',
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