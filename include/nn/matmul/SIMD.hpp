#ifndef NN_MATMUL_SIMD_HPP
#define NN_MATMUL_SIMD_HPP

#include <arm_neon.h>

#include "../Matrix.hpp"

namespace nn::matmul {

Matrix multiply_blocked_simd(const Matrix& A, const Matrix& B, int BLOCK_SIZE = 64);

}


#endif