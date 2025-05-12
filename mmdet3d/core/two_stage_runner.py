import time
import warnings

import mmcv
from mmcv.runner.utils import get_host_info
from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class TwoStageRunner(EpochBasedRunner):
    def __init__(self, first_stage_ratio=0.75, **kwargs):
        super(TwoStageRunner, self).__init__(**kwargs)
        self.switch_epoch = first_stage_ratio * self._max_epochs

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert len(data_loaders) == 2
        assert mmcv.is_list_of(workflow, tuple)
        assert len(workflow) == 1
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    data_loader_curr = data_loaders[0] if self.epoch<self.switch_epoch \
                        else data_loaders[1]
                    self.logger.info('Data processing pipeline of the current epoch:')
                    for tf in data_loader_curr.dataset.dataset.pipeline.transforms:
                        self.logger.info(type(tf).__name__)
                    self.logger.info('Start running, host: %s, work_dir: %s',
                                     get_host_info(), work_dir)
                    epoch_runner(data_loader_curr, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')