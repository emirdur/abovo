#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP
#include "Matrix.hpp"

class DenseLayer {
private:
    int input_size;
    int output_size;
    Matrix weights;
    Matrix biases;

public:
    DenseLayer(int in, int out);

    Matrix forward(const Matrix& X) const;
    Matrix activation(const Matrix& X) const;
    void print() const;
};

#endif