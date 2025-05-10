import sys
from unittest.mock import MagicMock

class Matrix(MagicMock):
    def get_rows(self): return 0
    def get_cols(self): return 0
    def transpose(self): return Matrix()
    def get_min(self): return 0.0
    def get_max(self): return 0.0
    def print(self): pass

class DenseLayer(MagicMock):
    def forward(self, input): return Matrix()
    def quantize(self): pass
    def dequantize(self): pass
    def is_quantized(self): return False
    def print(self): pass

class Sequential(MagicMock):
    def add(self, layer): pass
    def forward(self, X): return Matrix()
    def train(self, X, y, epochs, batch_size, learning_rate, loss_type=None): pass
    def evaluate(self, X, y): pass
    def quantize_all(self, per_channel=True): pass
    def dequantize_all(self): pass
    def enable_qat(self): pass
    def enable_adam(self, enable=True, beta1=0.9, beta2=0.999, epsilon=1e-8): pass
    def print(self): pass

class LossType:
    MSE = 0
    CROSS_ENTROPY = 1

class ActivationType:
    RELU = 0
    LEAKY_RELU = 1
    SIGMOID = 2
    SOFTMAX = 3

mock_abovo = MagicMock()
mock_abovo.Matrix = Matrix
mock_abovo.DenseLayer = DenseLayer
mock_abovo.Sequential = Sequential
mock_abovo.LossType = LossType
mock_abovo.ActivationType = ActivationType

sys.modules["_abovo"] = mock_abovo