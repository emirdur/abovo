#ifndef NN_MATMUL_HPP
#define NN_MATMUL_HPP

#include "Matrix.hpp"

namespace nn {

enum class MatMulType {
    NAIVE,
    BLOCKED,
    SIMD,
    SIMD_MT,
    METAL_GPU
};

class MatMul {
public:
    Matrix matrix_multiply(const Matrix& A, const Matrix& B, MatMulType type = MatMulType::NAIVE, int block_size = 64, int num_threads = 0);
};

}

#endif