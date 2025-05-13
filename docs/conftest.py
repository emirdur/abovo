import sys
from unittest.mock import MagicMock
from enum import IntEnum

class LossType(IntEnum):
    MSE = 0
    CROSS_ENTROPY = 1

class ActivationType(IntEnum):
    RELU = 0
    LEAKY_RELU = 1
    SIGMOID = 2
    SOFTMAX = 3

class MatMulType(IntEnum):
    NAIVE = 0
    BLOCKED = 1
    SIMD = 2
    SIMD_MT = 3
    METAL_GPU = 4

class MatrixMock(MagicMock):
    def get_rows(self): return 0
    def get_cols(self): return 0
    def transpose(self): return MatrixMock()
    def get_min(self): return 0.0
    def get_max(self): return 0.0
    def print(self): pass

class DenseLayerMock(MagicMock):
    def forward(self, input): return MatrixMock()
    def quantize(self): pass
    def dequantize(self): pass
    def is_quantized(self): return False
    def print(self): pass

class SequentialMock(MagicMock):
    def add(self, layer): pass
    def forward(self, X): return MatrixMock()
    def train(self, X, y, epochs, batch_size, learning_rate, loss_type=None): pass
    def evaluate(self, X, y): pass
    def quantize_all(self, per_channel=True): pass
    def dequantize_all(self): pass
    def enable_qat(self): pass
    def enable_adam(self, enable=True, beta1=0.9, beta2=0.999, epsilon=1e-8): pass
    def print(self): pass

sys.modules["_abovo"] = type('_abovo', (), {
    'Matrix': MatrixMock,
    'DenseLayer': DenseLayerMock,
    'Sequential': SequentialMock,
    'LossType': LossType,
    'ActivationType': ActivationType,
    'MatMulType': MatMulType,
})