# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DDetector
from .centerpoint import CenterPoint
from .mvx_two_stage import MVXTwoStageDetector

__all__ = [
    'Base3DDetector', 'MVXTwoStageDetector',
    'CenterPoint'
]
