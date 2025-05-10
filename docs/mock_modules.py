class Matrix:
    def __init__(self, *args, **kwargs):
        pass
    def get_rows(self):
        pass
    def get_cols(self):
        pass
    def transpose(self):
        pass
    def get_min(self):
        pass
    def get_max(self):
        pass
    def print(self):
        pass

class DenseLayer:
    def __init__(self, *args, **kwargs):
        pass
    def forward(self, input):
        pass
    def quantize(self):
        pass
    def dequantize(self):
        pass
    def is_quantized(self):
        pass
    def print(self):
        pass

class Sequential:
    def __init__(self):
        pass
    def add(self, layer):
        pass
    def forward(self, X):
        pass
    def train(self, X, y, epochs, batch_size, learning_rate, loss_type=None):
        pass
    def evaluate(self, X, y):
        pass
    def quantize_all(self, per_channel=True):
        pass
    def dequantize_all(self):
        pass
    def enable_qat(self):
        pass
    def enable_adam(self, enable=True, beta1=0.9, beta2=0.999, epsilon=1e-8):
        pass
    def print(self):
        pass

class LossType:
    MSE = 0
    CROSS_ENTROPY = 1

class ActivationType:
    RELU = 0
    LEAKY_RELU = 1
    SIGMOID = 2
    SOFTMAX = 3