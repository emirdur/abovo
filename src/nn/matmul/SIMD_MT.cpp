#include "nn/matmul/SIMD_MT.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <omp.h>

namespace nn::matmul {

    Matrix multiply_blocked_simd_mt(const Matrix& A, const Matrix& B, int BLOCK_SIZE, int num_threads) {
        if (A.getCols() != B.getRows()) {
            throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
        }
    
        int rows = A.getRows();
        int cols = B.getCols();
        int inner = A.getCols();

        num_threads = omp_get_max_threads();
    
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        
        Matrix res(rows, cols);
    
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                res(i, j) = 0.0;
            }
        }
        
        #pragma omp parallel
        {
            std::vector<float> A_block(BLOCK_SIZE * BLOCK_SIZE);
            std::vector<float> B_block(BLOCK_SIZE * BLOCK_SIZE);
            std::vector<float> C_block(BLOCK_SIZE * BLOCK_SIZE);
            
            #pragma omp for schedule(dynamic)
            for (int ii = 0; ii < rows; ii += BLOCK_SIZE) {
                for (int jj = 0; jj < cols; jj += BLOCK_SIZE) {
                    int block_rows = std::min(BLOCK_SIZE, rows - ii);
                    int block_cols = std::min(BLOCK_SIZE, cols - jj);
                    
                    std::fill(C_block.begin(), C_block.end(), 0.0);
                    
                    for (int kk = 0; kk < inner; kk += BLOCK_SIZE) {
                        int block_inner = std::min(BLOCK_SIZE, inner - kk);
                        
                        for (int i = 0; i < block_rows; ++i) {
                            for (int k = 0; k < block_inner; ++k) {
                                A_block[k * BLOCK_SIZE + i] = static_cast<float>(A(ii + i, kk + k));
                            }
                        }
                        
                        for (int k = 0; k < block_inner; ++k) {
                            for (int j = 0; j < block_cols; ++j) {
                                B_block[j * BLOCK_SIZE + k] = static_cast<float>(B(kk + k, jj + j));
                            }
                        }
                        
                        int simd_rows = (block_rows / 4) * 4;
                        int simd_cols = (block_cols / 4) * 4;
                        
                        for (int i = 0; i < simd_rows; i += 4) {
                            for (int j = 0; j < simd_cols; j += 4) {
                                float32x4_t A0;
                                float32x4_t C0, C1, C2, C3;
                                
                                C0 = vld1q_f32(&C_block[(j + 0) * BLOCK_SIZE + i]);
                                C1 = vld1q_f32(&C_block[(j + 1) * BLOCK_SIZE + i]);
                                C2 = vld1q_f32(&C_block[(j + 2) * BLOCK_SIZE + i]);
                                C3 = vld1q_f32(&C_block[(j + 3) * BLOCK_SIZE + i]);
                                
                                for (int k = 0; k < block_inner; ++k) {
                                    A0 = vld1q_f32(&A_block[k * BLOCK_SIZE + i]);
                                    
                                    float b0 = B_block[(j + 0) * BLOCK_SIZE + k];
                                    float b1 = B_block[(j + 1) * BLOCK_SIZE + k];
                                    float b2 = B_block[(j + 2) * BLOCK_SIZE + k];
                                    float b3 = B_block[(j + 3) * BLOCK_SIZE + k];
                                    
                                    C0 = vmlaq_n_f32(C0, A0, b0);
                                    C1 = vmlaq_n_f32(C1, A0, b1);
                                    C2 = vmlaq_n_f32(C2, A0, b2);
                                    C3 = vmlaq_n_f32(C3, A0, b3);
                                }
                                
                                vst1q_f32(&C_block[(j + 0) * BLOCK_SIZE + i], C0);
                                vst1q_f32(&C_block[(j + 1) * BLOCK_SIZE + i], C1);
                                vst1q_f32(&C_block[(j + 2) * BLOCK_SIZE + i], C2);
                                vst1q_f32(&C_block[(j + 3) * BLOCK_SIZE + i], C3);
                            }
                        }
                        
                        for (int i = 0; i < block_rows; ++i) {
                            for (int j = simd_cols; j < block_cols; ++j) {
                                for (int k = 0; k < block_inner; ++k) {
                                    C_block[j * BLOCK_SIZE + i] += A_block[k * BLOCK_SIZE + i] * B_block[j * BLOCK_SIZE + k];
                                }
                            }
                        }
                        
                        for (int i = simd_rows; i < block_rows; ++i) {
                            for (int j = 0; j < block_cols; ++j) {
                                for (int k = 0; k < block_inner; ++k) {
                                    C_block[j * BLOCK_SIZE + i] += A_block[k * BLOCK_SIZE + i] * B_block[j * BLOCK_SIZE + k];
                                }
                            }
                        }
                    }
  
                    for (int i = 0; i < block_rows; ++i) {
                        for (int j = 0; j < block_cols; ++j) {
                            res(ii + i, jj + j) = static_cast<double>(C_block[j * BLOCK_SIZE + i]);
                        }
                    }
                }
            }
        }
        
        return res;
    }
}