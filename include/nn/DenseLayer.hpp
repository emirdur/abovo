#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include "Matrix.hpp"
#include "nn/quantization/Int8Matrix.hpp"
#include "Activation.hpp"

namespace nn {

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

    bool is_quantized = false;
    quantization::Int8Matrix quantized_weights;

public:
    DenseLayer(int in, int out, ActivationType activation_type);

    Matrix forward(const Matrix& X);
    Matrix activation(const Matrix& X) const;
    void print() const;

    Matrix backward(const Matrix& d_out, double eta);

    void quantize();
    void dequantize();
    bool isQuantized() const;
};

}

#endif