#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "nn/Activation.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Matrix.hpp"
#include "nn/Sequential.hpp"
#include "nn/matmul/Blocked.hpp"
#include "nn/matmul/Naive.hpp"
#include "nn/matmul/SIMD.hpp"
#include "nn/matmul/SIMD_MT.hpp"

using namespace nn;

int main() {
  // Playground

  // You can test whatever you'd like in here. The Dockerfile is pre-written to
  // run valgrind to check for cache misses. For example, you can test out how
  // many cache misses the naive matrix multiplication has:

  int MATRIX_SIZE = 256;

  Matrix A(MATRIX_SIZE, MATRIX_SIZE);
  Matrix B(MATRIX_SIZE, MATRIX_SIZE);
  A.randomize(MATRIX_SIZE);
  B.randomize(MATRIX_SIZE);

  Matrix C = matmul::multiply_blocked_simd_mt(A, B);

  // Once you run your Dockerfile (docker build -t nn-ab-ovo .; docker run --rm
  // nn-ab-ovo) you'll be able to profile cache misses and compare it with the
  // blocked multiplication version.
  return 0;
}