#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "DenseLayer.hpp"


DenseLayer::DenseLayer(int in, int out): input_size(in), output_size(out), weights(input_size, output_size), biases(1, output_size) {	
	weights.randomize();
	
	for (int j = 0; j < output_size; ++j) {
		biases(0, j) = 0.0;
	}
}

Matrix DenseLayer::forward(const Matrix& X) const {
	return X * weights + biases;
}

// ReLU for now
Matrix DenseLayer::activation(const Matrix& X) const {
	int X_rows = X.getRows();
	int X_cols = X.getCols();

	Matrix res(X_rows, X_cols);
	for (int i = 0; i < X_rows; ++i) {
		for (int j = 0; j < X_cols; ++j) {
			res(i, j) = std::max(0.0, X(i, j));
		}
	}

	return res;
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
