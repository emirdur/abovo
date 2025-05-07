#ifndef NN_MATMUL_SIMD_MT_HPP
#define NN_MATMUL_SIMD_MT_HPP

#include <arm_neon.h>
#include <omp.h>

#include "../Matrix.hpp"

namespace nn::matmul {

Matrix multiply_blocked_simd_mt(const Matrix& A, const Matrix& B, int BLOCK_SIZE=64, int num_threads=0);

}


#endif