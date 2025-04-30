#include "Activation.hpp"
#include <cmath>
#include <algorithm>

Matrix Activation::relu(const Matrix& X) {
    Matrix res(X.getRows(), X.getCols());
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            res(i, j) = std::max(0.0, X(i, j));
        }
    }
    return res;
}

Matrix Activation::relu_derivative(const Matrix& X) {
    Matrix res(X.getRows(), X.getCols());
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            res(i, j) = X(i, j) > 0 ? 1.0 : 0.0;
        }
    }
    return res;
}

Matrix Activation::leaky_relu(const Matrix& X, double alpha) {
    Matrix res(X.getRows(), X.getCols());
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            res(i, j) = X(i, j) > 0 ? X(i, j) : alpha * X(i, j);
        }
    }
    return res;
}

Matrix Activation::leaky_relu_derivative(const Matrix& X, double alpha) {
    Matrix res(X.getRows(), X.getCols());
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            res(i, j) = X(i, j) > 0 ? 1.0 : alpha;
        }
    }
    return res;
}

Matrix Activation::sigmoid(const Matrix& X) {
    Matrix res(X.getRows(), X.getCols());
    
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            double x = X(i, j);
            res(i, j) = 1.0 / (1.0 + std::exp(-x));
        }
    }
    
    return res;
}

Matrix Activation::sigmoid_derivative(const Matrix& X) {
    Matrix res(X.getRows(), X.getCols());
    
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            double sigmoid_x = 1.0 / (1.0 + std::exp(-X(i, j)));
            
            res(i, j) = sigmoid_x * (1.0 - sigmoid_x);
        }
    }
    
    return res;
}