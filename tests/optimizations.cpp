#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

#include "nn/Matrix.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Sequential.hpp"
#include "nn/Activation.hpp"
#include "nn/matmul/Naive.hpp"
#include "nn/matmul/Blocked.hpp"
#include "nn/matmul/SIMD.hpp"

using namespace nn;

// get all times and sum the start times together, then the end times together and take the average
void printAverage(const std::vector<long long>& times, const std::string& operation) {
    if (times.empty()) { 
        return;
    }
    
    long long sum = std::accumulate(times.begin(), times.end(), 0LL); // value 0 type long long
    double avg = static_cast<double>(sum) / times.size();
    
    std::cout << operation << " Average: " << avg << " µs." << std::endl;
}

int main() {
    const int MATRIX_SIZE = 256;
    const int NUM_ITERATIONS = 10;
    
    std::cout << "Matrix Multiplication Benchmark" << std::endl;
    std::cout << "Matrix Size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
    
    Matrix A(MATRIX_SIZE, MATRIX_SIZE);
    Matrix B(MATRIX_SIZE, MATRIX_SIZE);
    A.randomize(MATRIX_SIZE);
    B.randomize(MATRIX_SIZE);
    
    std::vector<long long> multiplyTimes;
    multiplyTimes.reserve(NUM_ITERATIONS);

    for (int w = 0; w < 3; w++) {
        Matrix C = matmul::multiply_naive(A, B);
    }
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Matrix C = matmul::multiply_naive(A, B);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        multiplyTimes.push_back(duration.count());
        std::cout << "Iteration " << (i+1) << ": " << duration.count() << " µs" << std::endl;
    }

    printAverage(multiplyTimes, "Matrix Multiplication");

    multiplyTimes.clear();

    std::cout << std::endl << "Matrix Multiplication (Blocking)" << std::endl;
    std::cout << "Matrix Size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;

    for (int w = 0; w < 3; w++) {
        Matrix C = matmul::multiply_blocked(A, B);
    }
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Matrix C = matmul::multiply_blocked(A, B);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        multiplyTimes.push_back(duration.count());
        std::cout << "Iteration " << (i+1) << ": " << duration.count() << " µs" << std::endl;
    }

    printAverage(multiplyTimes, "Matrix Multiplication (Blocking)");

    multiplyTimes.clear();

    std::cout << std::endl << "Matrix Multiplication (SIMD Blocking)" << std::endl;
    std::cout << "Matrix Size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;

    for (int w = 0; w < 3; w++) {
        Matrix C = matmul::multiply_blocked_simd(A, B);
    }
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Matrix C = matmul::multiply_blocked_simd(A, B);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        multiplyTimes.push_back(duration.count());
        std::cout << "Iteration " << (i+1) << ": " << duration.count() << " µs" << std::endl;
    }

    printAverage(multiplyTimes, "Matrix Multiplication (SIMD Blocking)");
    
    return 0;
}