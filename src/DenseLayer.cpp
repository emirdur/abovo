#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "DenseLayer.hpp"
#include "Activation.hpp"


DenseLayer::DenseLayer(int in, int out): input_size(in), output_size(out), weights(input_size, output_size), biases(1, output_size) {	
	weights.randomize(input_size);
	
	for (int j = 0; j < output_size; ++j) {
		biases(0, j) = 0.0;
	}
}

Matrix DenseLayer::forward(const Matrix& X) {
	last_input = X;
	Matrix z = X * weights + biases;
	last_linear_output = z;
	return z;
}

// ReLU for now
Matrix DenseLayer::activation(const Matrix& X) const {
	return Activation::leaky_relu(X);
}

void DenseLayer::print() const {
	std::cout << "Weights:" << std::endl;
	for (int i = 0; i < input_size; ++i) {
		for (int j = 0; j < output_size; ++j) {
			std::cout << weights(i, j) << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl << "Biases" << std::endl;
	for (int i = 0; i < output_size; ++i) {
		std::cout << biases(0, i) << " ";
	}

	std::cout << std::endl;
}

Matrix DenseLayer::backward(const Matrix& incoming_gradient, double learning_rate) {
	// z = a * W + b
	// incoming_gradient = dC_0 / da^(L)
	// last_linear_output.relu_derivative() = da^(L) / dz^(L)
	// we hadamard product it because each gradient requires an activation derivative
	// adjusted_gradient = dC_0 / da^(L) * da^(L) / dz^(L) = dC_0 / dz^(L)
	Matrix adjusted_gradient = incoming_gradient.hadamard_product(Activation::leaky_relu_derivative(last_linear_output));

	// check for explosions
	double max_val = -1e9;
	double min_val = 1e9;

	for (int i = 0; i < adjusted_gradient.getRows(); ++i) {
		for (int j = 0; j < adjusted_gradient.getCols(); ++j) {
			max_val = std::max(max_val, adjusted_gradient(i, j));
			min_val = std::min(min_val, adjusted_gradient(i, j));
		}
	}

	if (std::isnan(max_val) || std::isnan(min_val)) {
		std::cerr << "Explosion detected in adjusted gradient" << std::endl;
	}
	else if (std::abs(max_val) > 1e3 || std::abs(min_val) > 1e3) {
		std::cerr << "Warning: Large gradient values detected. Max: " << max_val << " Min: " << min_val << std::endl;
	}

	// last_input.tranpose() = dz^(L) / dw^(L)
	// grad_weights = dC_0 / dw^(L) = chain rule from other influences
	Matrix grad_weights = last_input.transpose() * adjusted_gradient;

	int adjusted_gradient_rows = adjusted_gradient.getRows();
	int adjusted_gradient_cols = adjusted_gradient.getCols();

	// dC_0 / db = dC_0 / dz * dz / db = dC_0 / dz 
	Matrix grad_biases(1, adjusted_gradient_cols);
    for (int j = 0; j < adjusted_gradient_cols; ++j) {
        double sum = 0.0;
        for (int i = 0; i < adjusted_gradient_rows; ++i) {
            sum += adjusted_gradient(i, j);
        }
        grad_biases(0, j) = sum;
    }

	weights = weights + ((grad_weights * learning_rate) * -1);
    biases = biases + ((grad_biases * learning_rate) * -1);

	// pass gradient to previous layer by multiplying current gradient with weights: dC_0/dz^(L) * dz^(L)/da^(L-1) = dC_0/da^(L-1)
    return adjusted_gradient * weights.transpose();
}

