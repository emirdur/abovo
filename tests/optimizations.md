# Optimization Experiments

This document tracks optimization experiments performed on a C++ neural network engine. Each experiment evaluates the performance of a current baseline model, with a focus on understanding the systems-level impact of specific optimizations. We use this to profile and document optimizations done to our neural network.

---

## Experiment 1: Naive Matrix Multiplication vs. Blocked Matrix Multiplication

### Setup

**Hardware**: Apple M2 Pro (Docker running x86_64 emulated via Debian)  
**Compiler**: `g++`  
**Dataset**: XOR binary classification  
**Model**:

```cpp
model.add(DenseLayer(2, 8, ActivationType::LEAKY_RELU));
model.add(DenseLayer(8, 1, ActivationType::SIGMOID));
model.train(X, y, 1000, 2, 0.1);
```

**Matrix Multiplication**: Naive triple-loop  
**Compiler Flags**: -O2  
**Timing**: std::chrono  
**Profiling Tool**: valgrind --tool=cachegrind  
**Goal**: Establish base cache behavior and instruction count

### Results

| Metric                 | Naive   | Blocked | Improvement              |
| ---------------------- | ------- | ------- | ------------------------ |
| Instructions Executed  | 438M    | 95M     | ~4.6× fewer instructions |
| L1 Data Cache Misses   | 213,564 | 46,152  | ~4.6× fewer misses       |
| LL Data Cache Misses   | 10,597  | 10,488  | Similar                  |
| Total LL Cache Misses  | 12,610  | 12,490  | Slight improvement       |
| D Refs (Data Accesses) | 199M    | 43M     | ~4.6× fewer accesses     |

| Implementation | Avg Time (µs) | Speedup |
| -------------- | ------------- | ------- |
| Naive          | 17,706.1      | –       |
| Blocking       | 2,987.1       | ~5.93×  |
