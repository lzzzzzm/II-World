# Copyright (c) OpenMMLab. All rights reserved.
import platform
from functools import partial

from mmcv.utils import Registry, build_from_cfg
from mmcv.parallel import collate
from mmcv.runner import get_dist_info

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset, worker_init_fn
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import (DistributedGroupSampler,
                       DistributedSampler, GroupSampler)

from .samplers import InfiniteGroupEachSampleInBatchSampler, CustomDistributedSampler, InfiniteGroupEachSampleInBatchSamplerEval, TTADistributedSampler, InfiniteGroupEachSampleInBatchSamplerWeighted

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     val=False,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
    if val:
        # runner_type = 'EpochBasedRunner'
        assert not shuffle
    if runner_type == 'IterBasedRunner':
        # TODO: original has more options, but I'm not using them
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157

        batch_sampler = InfiniteGroupEachSampleInBatchSampler(
            dataset,
            batch_size,
            world_size,
            rank,
            seed=seed)
        # batch_sampler = InfiniteGroupEachSampleInBatchSamplerWeighted(
        #     dataset,
        #     batch_size,
        #     world_size,
        #     rank,
        #     seed=seed)
        batch_size = 1
        sampler = None

    elif runner_type == 'IterBasedRunnerEval':
        # TODO: original has more options, but I'm not using them
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157

        batch_sampler = InfiniteGroupEachSampleInBatchSamplerEval(
            dataset,
            batch_size,
            world_size,
            rank,
            seed=seed)
        batch_size = 1
        sampler = None
    elif runner_type == 'TTARunnerEval':
        # TODO: original has more options, but I'm not using them
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157

        batch_sampler = TTADistributedSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed)
        sampler = None
    else:
        if dist:
            # DistributedGroupSampler will definitely shuffle the data to satisfy
            # that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                if val:
                    sampler = CustomDistributedSampler(
                        dataset, world_size, rank, shuffle=False, seed=seed)
                else:
                    sampler = DistributedSampler(
                        dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None

        batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader