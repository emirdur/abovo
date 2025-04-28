#ifndef LOSS_HPP
#define LOSS_HPP
#include "Matrix.hpp"

class Loss {
public:
    double mse(const Matrix& y_pred, const Matrix& y_true) const;
    Matrix d_mse(const Matrix& y_pred, const Matrix& y_true) const;
};
#endif