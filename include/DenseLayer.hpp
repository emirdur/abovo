#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP
#include "Matrix.hpp"
#include "Activation.hpp"

class DenseLayer {
private:
    int input_size;
    int output_size;
    Matrix weights;
    Matrix biases;

    Matrix last_input;
    // where z = w * x + b
    Matrix last_linear_output;

    ActivationType activation_type;

public:
    DenseLayer(int in, int out, ActivationType activation_type);

    Matrix forward(const Matrix& X);
    Matrix activation(const Matrix& X) const;
    void print() const;

    Matrix backward(const Matrix& d_out, double eta);
};

#endif