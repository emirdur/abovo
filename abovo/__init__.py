"""
Abovo: A C++ neural network engine with Python bindings for educational performance optimization.
"""

__version__ = "0.1.0"

from _abovo import (
    Matrix,
    DenseLayer,
    Sequential,
    ActivationType,
    LossType
)

__all__ = [
    'Matrix',
    'DenseLayer',
    'Sequential',
    'ActivationType',
    'LossType'
]