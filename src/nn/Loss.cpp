#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include "nn/Loss.hpp"
#include "nn/loss/MSE.hpp"

namespace nn {

double Loss::loss(const Matrix& y_pred, const Matrix& y_true) const {
    return loss::mse(y_pred, y_true);
}

Matrix Loss::loss_derivative(const Matrix& y_pred, const Matrix& y_true) const {
    return loss::mse_derivative(y_pred, y_true);
}

}