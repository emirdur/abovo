#ifndef LOSS_HPP
#define LOSS_HPP

#include "Matrix.hpp"

namespace nn {

class Loss {
public:
    double loss(const Matrix& y_pred, const Matrix& y_true) const;
    Matrix loss_derivative(const Matrix& y_pred, const Matrix& y_true) const;
};

}
#endif