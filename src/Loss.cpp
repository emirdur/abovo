#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include "Loss.hpp"

double Loss::mse(const Matrix& y_pred, const Matrix& y_true) const {
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    int rows = y_pred.getRows();
    int cols = y_pred.getCols();
    double sum = 0.0;
    int n = rows * cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff = y_true(i, j) - y_pred(i, j);
            sum += diff * diff;
        }
    }

    return sum / n;
}

Matrix Loss::d_mse(const Matrix& y_pred, const Matrix& y_true) const {
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    int rows = y_pred.getRows();
    int cols = y_pred.getCols();
    int n = rows * cols;

    Matrix grad(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grad(i, j) = 2.00 * (y_true(i, j) - y_pred(i, j)) / n;
        }
    }

    return grad;
}