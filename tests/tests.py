import unittest
import numpy as np
import _abovo
import tempfile
import os

class TestMatrix(unittest.TestCase):
    def test_matrix_creation(self):
        # Test creation with dimensions
        matrix = _abovo.Matrix(3, 4)
        self.assertEqual(matrix.get_rows(), 3)
        self.assertEqual(matrix.get_cols(), 4)
        
        # Test creation with data
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix = _abovo.Matrix(data)
        self.assertEqual(matrix.get_rows(), 2)
        self.assertEqual(matrix.get_cols(), 3)
        
    def test_matrix_access(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix = _abovo.Matrix(data)
        
        # Test getitem
        self.assertEqual(matrix[0, 0], 1.0)
        self.assertEqual(matrix[0, 2], 3.0)
        self.assertEqual(matrix[1, 1], 5.0)
        
        # Test setitem
        matrix[0, 0] = 10.0
        self.assertEqual(matrix[0, 0], 10.0)
        
    def test_matrix_operations(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix = _abovo.Matrix(data)
        
        # Test transpose
        transposed = matrix.transpose()
        self.assertEqual(transposed.get_rows(), 3)
        self.assertEqual(transposed.get_cols(), 2)
        self.assertEqual(transposed[0, 0], 1.0)
        self.assertEqual(transposed[0, 1], 4.0)
        self.assertEqual(transposed[1, 0], 2.0)
        
        # Test min/max
        self.assertEqual(matrix.get_min(), 1.0)
        self.assertEqual(matrix.get_max(), 6.0)
        
    def test_matrix_print(self):
        matrix = _abovo.Matrix(2, 2)
        # Just test that print doesn't crash
        matrix.print()


class TestDenseLayer(unittest.TestCase):
    def test_layer_creation(self):
        # Test with different activation functions
        layer_relu = _abovo.DenseLayer(10, 5, _abovo.ActivationType.RELU)
        layer_sigmoid = _abovo.DenseLayer(10, 5, _abovo.ActivationType.SIGMOID)
        layer_softmax = _abovo.DenseLayer(10, 5, _abovo.ActivationType.SOFTMAX)
        
        self.assertIsNotNone(layer_relu)
        self.assertIsNotNone(layer_sigmoid)
        self.assertIsNotNone(layer_softmax)
        
    def test_layer_forward(self):
        layer = _abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU)
        
        input_data = [[1.0, 2.0, 3.0]]
        input_matrix = _abovo.Matrix(input_data)
        
        # Test forward pass
        output = layer.forward(input_matrix)
        self.assertEqual(output.get_rows(), 1)
        self.assertEqual(output.get_cols(), 2)
        
    def test_layer_quantization(self):
        layer = _abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU)
        
        # Test quantization
        self.assertFalse(layer.is_quantized())
        layer.quantize(True)
        self.assertTrue(layer.is_quantized())
        
        # Test dequantization
        layer.dequantize()
        self.assertFalse(layer.is_quantized())
        
    def test_layer_print(self):
        layer = _abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU)
        layer.print()


class TestSequential(unittest.TestCase):
    def test_sequential_creation(self):
        model = _abovo.Sequential()
        self.assertIsNotNone(model)
        
    def test_sequential_add_layer(self):
        model = _abovo.Sequential()
        layer1 = _abovo.DenseLayer(10, 5, _abovo.ActivationType.RELU)
        layer2 = _abovo.DenseLayer(5, 2, _abovo.ActivationType.SOFTMAX)
        
        model.add(layer1)
        model.add(layer2)
        
        model.print()
        
    def test_sequential_forward(self):
        model = _abovo.Sequential()
        model.add(_abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU))
        model.add(_abovo.DenseLayer(2, 1, _abovo.ActivationType.SIGMOID))
        
        input_data = [[1.0, 2.0, 3.0]]
        input_matrix = _abovo.Matrix(input_data)
        
        # Test forward pass
        output = model.forward(input_matrix)
        self.assertEqual(output.get_rows(), 1)
        self.assertEqual(output.get_cols(), 1)
        
    def test_sequential_train(self):
        model = _abovo.Sequential()
        model.add(_abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU))
        model.add(_abovo.DenseLayer(2, 1, _abovo.ActivationType.SIGMOID))
        
        X_data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ]
        y_data = [
            [0.0],
            [1.0],
            [0.0],
            [1.0]
        ]
        
        X = _abovo.Matrix(X_data)
        y = _abovo.Matrix(y_data)
        
        model.train(X, y, epochs=10, batch_size=2, learning_rate=0.01, loss_type=_abovo.LossType.MSE)
        
        # Test evaluation
        loss = model.evaluate(X, y)
        self.assertIsInstance(loss, float)
        
    def test_adam_optimizer(self):
        model = _abovo.Sequential()
        model.add(_abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU))
        model.add(_abovo.DenseLayer(2, 1, _abovo.ActivationType.SIGMOID))
        
        model.enable_adam(True, 0.9, 0.999, 1e-8)
        
        X = _abovo.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = _abovo.Matrix([[0.0], [1.0]])
        
        model.train(X, y, epochs=5, batch_size=1, learning_rate=0.001)
        
        model.enable_adam(False)
        
    def test_qat(self):
        model = _abovo.Sequential()
        model.add(_abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU))
        model.add(_abovo.DenseLayer(2, 1, _abovo.ActivationType.SIGMOID))
        
        model.enable_qat(True)
        
        X = _abovo.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = _abovo.Matrix([[0.0], [1.0]])
        
        # Train with QAT
        model.train(X, y, epochs=5, batch_size=1, learning_rate=0.01)
        
    def test_quantization(self):
        model = _abovo.Sequential()
        model.add(_abovo.DenseLayer(3, 2, _abovo.ActivationType.RELU))
        model.add(_abovo.DenseLayer(2, 1, _abovo.ActivationType.SIGMOID))
        
        # Test quantization
        model.quantize_all(True) 
        
        input_data = [[1.0, 2.0, 3.0]]
        input_matrix = _abovo.Matrix(input_data)

        output = model.forward(input_matrix)
        self.assertEqual(output.get_rows(), 1)
        self.assertEqual(output.get_cols(), 1)
        
        # Test dequantization
        model.dequantize_all()


class TestLossType(unittest.TestCase):
    def test_loss_type_enum(self):
        # Test loss type enum values
        self.assertEqual(_abovo.LossType.MSE, _abovo.LossType.MSE)
        self.assertEqual(_abovo.LossType.CrossEntropy, _abovo.LossType.CrossEntropy)
        self.assertNotEqual(_abovo.LossType.MSE, _abovo.LossType.CrossEntropy)


class TestActivationType(unittest.TestCase):
    def test_activation_type_enum(self):
        # Test activation type enum values
        self.assertEqual(_abovo.ActivationType.RELU, _abovo.ActivationType.RELU)
        self.assertEqual(_abovo.ActivationType.SIGMOID, _abovo.ActivationType.SIGMOID)
        self.assertEqual(_abovo.ActivationType.SOFTMAX, _abovo.ActivationType.SOFTMAX)
        self.assertEqual(_abovo.ActivationType.LEAKY_RELU, _abovo.ActivationType.LEAKY_RELU)
        self.assertNotEqual(_abovo.ActivationType.RELU, _abovo.ActivationType.SIGMOID)


class TestIntegration(unittest.TestCase):
    def test_xor_problem(self):
        model = _abovo.Sequential()
        model.add(_abovo.DenseLayer(2, 8, _abovo.ActivationType.RELU))
        model.add(_abovo.DenseLayer(8, 1, _abovo.ActivationType.SIGMOID))
        
        X_data = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
        y_data = [
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ]
        
        X = _abovo.Matrix(X_data)
        y = _abovo.Matrix(y_data)
        
        model.enable_adam(True, 0.9, 0.999, 1e-8)
        
        model.train(X, y, epochs=2000, batch_size=4, learning_rate=0.02)
        
        final_loss = model.evaluate(X, y)

        predictions_correct = True
        expected_outputs = [[0.0], [1.0], [1.0], [0.0]]
        
        for i, x in enumerate(X_data):
            input_matrix = _abovo.Matrix([x])
            output = model.forward(input_matrix)
            predicted = output[0, 0]
            expected = expected_outputs[i][0]
            
            if (predicted > 0.5 and expected < 0.5) or (predicted < 0.5 and expected > 0.5):
                predictions_correct = False
                break
        
        self.assertTrue(predictions_correct, 
                       f"XOR predictions incorrect. Final loss: {final_loss}")
        
        # Test predictions
        for i, x in enumerate(X_data):
            input_matrix = _abovo.Matrix([x])
            output = model.forward(input_matrix)
            
            predicted = output[0, 0]
            expected = y_data[i][0]
            
            if expected == 0.0:
                self.assertLess(predicted, 0.3)
            else:
                self.assertGreater(predicted, 0.7)


if __name__ == '__main__':
    unittest.main()