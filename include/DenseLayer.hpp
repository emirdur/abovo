#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP
#include "Matrix.hpp"

class DenseLayer {
private:
    int input_size;
    int output_size;
    Matrix weights;
    Matrix biases;

    Matrix last_input;
    // where z = w * x + b
    Matrix last_linear_output;

public:
    DenseLayer(int in, int out);

    Matrix forward(const Matrix& X);
    Matrix activation(const Matrix& X) const;
    void print() const;

    Matrix backward(const Matrix& d_out, double eta);
};

#endif