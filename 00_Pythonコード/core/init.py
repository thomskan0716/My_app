"""
Core modules for regression pipeline
"""

from .augmentation import PPMNoiseAugmentor
from .preprocessing import EnhancedPreprocessor, AdvancedFeatureSelector
from .utils import (
    fix_seed,
    choose_transform,
    apply_transform,
    inverse_transform,
    calculate_metrics,
    detect_outliers
)

__all__ = [
    'PPMNoiseAugmentor',
    'EnhancedPreprocessor',
    'AdvancedFeatureSelector',
    'fix_seed',
    'choose_transform',
    'apply_transform',
    'inverse_transform',
    'calculate_metrics',
    'detect_outliers'
]

__version__ = '1.0.0'