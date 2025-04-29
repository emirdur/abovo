#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Matrix.hpp"

enum class ActivationType {
    RELU,
    LEAKY_RELU,
};

class Activation {
public:
    static Matrix relu(const Matrix& X);
    static Matrix relu_derivative(const Matrix& X);

    static Matrix leaky_relu(const Matrix& X, double alpha = 0.01);
    static Matrix leaky_relu_derivative(const Matrix& X, double alpha = 0.01);
};

#endif