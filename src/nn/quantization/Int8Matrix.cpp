#include "nn/quantization/Int8Matrix.hpp"
#include <algorithm>

namespace nn::quantization {

Int8Matrix::Int8Matrix(size_t rows, size_t cols, float scale, float min): rows(rows), cols(cols), data(rows * cols), scale(scale), min(min) {}

int8_t& Int8Matrix::operator()(size_t i, size_t j) {
    return data[i * cols + j];
}

const int8_t& Int8Matrix::operator()(size_t i, size_t j) const {
    return data[i * cols + j];
}

Int8Matrix Int8Matrix::quantize(const nn::Matrix& X) {
    size_t rows = X.getRows();
    size_t cols = X.getCols();
    float min = static_cast<float>(X.getMin());
    float max = static_cast<float>(X.getMax());
    float scale = (max - min) / 255.0f;

    Int8Matrix res(rows, cols, scale, min);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res(i, j) = static_cast<int8_t>((X(i, j) - min) / scale);
        }
    }

    return res;
}

nn::Matrix Int8Matrix::dequantize() const {
    nn::Matrix res(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            res(i, j) = operator()(i, j) * scale + min;
        }
    }

    return res;
}

float Int8Matrix::getScale() const { 
    return scale; 
}

float Int8Matrix::getMin() const { 
    return min; 
}

}