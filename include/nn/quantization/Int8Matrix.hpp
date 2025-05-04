#ifndef NN_INT8MATRIX_HPP
#define NN_INT8MATRIX_HPP

#include <vector>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "../Matrix.hpp"

namespace nn::quantization { 

class Int8Matrix {
public:
    size_t rows;
    size_t cols;
    std::vector<int8_t> data;
    float scale;
    float min;

    Int8Matrix(size_t rows, size_t cols, float scale, float min);

    int8_t& operator()(size_t i, size_t j);
    const int8_t& operator()(size_t i, size_t j) const;

    static Int8Matrix quantize(const nn::Matrix& X);
    nn::Matrix dequantize() const;

    float getScale() const;
    float getMin() const;
};

} 

#endif