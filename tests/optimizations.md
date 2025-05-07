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
model.train(X, y, 1000, 2, 0.1, LossType::MSE);
```

A \* B, where A and B are 256x256 dimension matrices.

**Matrix Multiplication**: Naive triple-loop and multi-loop  
**Compiler Flags**: -O2  
**Timing**: std::chrono  
**Profiling Tool**: valgrind --tool=cachegrind  
**Goal**: Establish base cache behavior and instruction count as well as matrix multiplication speed.

### Results

| Metric                 | Naive   | Blocked    | Improvement              |
| ---------------------- | ------- | ---------- | ------------------------ |
| Instructions Executed  | 438M    | **95M**    | ~4.6× fewer instructions |
| L1 Data Cache Misses   | 213,564 | **46,152** | ~4.6× fewer misses       |
| LL Data Cache Misses   | 10,597  | **10,488** | Similar                  |
| Total LL Cache Misses  | 12,610  | **12,490** | Slight improvement       |
| D Refs (Data Accesses) | 199M    | **43M**    | ~4.6× fewer accesses     |

| Implementation | Avg Time (µs) | Speedup    |
| -------------- | ------------- | ---------- |
| Naive          | 17,706.1      | –          |
| **Blocking**   | **2,987.1**   | **~5.93×** |

## Experiment 2: SIMD Blocked Matrix Multiplication

### Setup

**Hardware**: Apple M2 Pro (Docker running x86_64 emulated via Debian)  
**Compiler**: `g++`  
**Dataset**: N/A  
**Model**:

A \* B, where A and B are 256x256 dimension matrices.

**Matrix Multiplication**: Multi-loop  
**Compiler Flags**: -O3 -ffast-math  
**Timing**: std::chrono  
**Profiling Tool**: valgrind --tool=cachegrind  
**Goal**: Compare SIMD Blocked Matrix Multiplication with Naive and Basic Blocked Matrix Multiplication.

### Results

| **Metric**                         | **Naive**     | **Blocked**  | **SIMD**        | **Notes / Improvement**                                                    |
| ---------------------------------- | ------------- | ------------ | --------------- | -------------------------------------------------------------------------- |
| **Instructions Executed (I refs)** | 137M          | 156M         | **46M**         | SIMD executes ~3× fewer instructions than Blocked and Naive                |
| **L1 I-Cache Misses (I1)**         | 1,838         | 1,839        | **1,868**       | All near 0% — instruction cache not a bottleneck                           |
| **LL I-Cache Misses (LLi)**        | 1,570         | 1,574        | **1,601**       | Negligible differences                                                     |
| **Data Refs (D refs)**             | 53.8M         | 71.3M        | **14.3M**       | SIMD uses ~3.7× fewer data accesses than Naive                             |
| **L1 D-Cache Misses (D1)**         | 17.8M (33.1%) | 2.37M (3.3%) | **967K (6.8%**) | Blocked significantly improves locality; SIMD benefits from fewer accesses |
| **LL D-Cache Misses (LLd)**        | 2.16M (4.0%)  | 110K (0.2%)  | **114K (0.8%)** | Blocked has best cache efficiency; SIMD is close behind                    |
| **Total LL Cache Misses**          | 2.16M         | 111K         | **116K**        | Blocked and SIMD reduce long-latency memory access significantly           |

> **Main Takeaway:** Blocking significantly reduces cache misses due to better spatial and temporal locality. SIMD further reduces instruction count and improves throughput with vectorized operations.

| Implementation | Avg Time (µs) | Speedup     |
| -------------- | ------------- | ----------- |
| Naive          | 11,039.1      | –           |
| Blocking       | 2,697.3       | ~4.09×      |
| **SIMD**       | **1,094.1**   | **~10.09×** |

## Experiment 3: SIMD Blocked Matrix Multiplication + Multi-Threading

### Setup

**Hardware**: Apple M2 Pro (Docker running x86_64 emulated via Debian)  
**Compiler**: `g++`  
**Dataset**: N/A  
**Model**:

A \* B, where A and B are 256x256 dimension matrices.

**Matrix Multiplication**: Multi-loop  
**Compiler Flags**: -O3 -ffast-math -fopenmp
**Timing**: std::chrono  
**Profiling Tool**: valgrind --tool=cachegrind  
**Goal**: Compare SIMD Blocked with Multi-Threading Matrix Multiplication to SIMD, Naive, and Basic Blocked Matrix Multiplication.

### Results

| **Metric**                     | **Naive**     | **Blocked**  | **SIMD**    | **SIMD + Multi-threaded** | **Notes / Improvement**                          |
| ------------------------------ | ------------- | ------------ | ----------- | ------------------------- | ------------------------------------------------ |
| Instructions Executed (I refs) | 137M          | 156M         | 46M         | **99.6M**                 | SIMD + threading reintroduces some overhead      |
| L1 I-Cache Misses (I1)         | 1,838         | 1,839        | 1,868       | **2,636**                 | Still <0.01% — not a bottleneck                  |
| LL I-Cache Misses (LLi)        | 1,570         | 1,574        | 1,601       | **2,189**                 | Negligible                                       |
| Data Refs (D refs)             | 53.8M         | 71.3M        | 14.3M       | **19.8M**                 | Increased due to parallel writes                 |
| L1 D-Cache Misses (D1)         | 17.8M (33.1%) | 2.37M (3.3%) | 967K (6.8%) | **948K (4.8%)**           | Comparable to SIMD; threads use more working set |
| LL D-Cache Misses (LLd)        | 2.16M (4.0%)  | 110K (0.2%)  | 114K (0.8%) | **125K (0.6%)**           | Still low — good locality retained               |
| **Total LL Cache Misses**      | 2.16M         | 111K         | 116K        | **127K**                  | Excellent performance across all optimizations   |

| **Implementation**        | **Avg Time (µs)** | **Speedup**  |
| ------------------------- | ----------------- | ------------ |
| Naive                     | 11,039.1          | –            |
| Blocking                  | 2,697.3           | \~4.09×      |
| SIMD                      | 1,094.1           | \~10.09×     |
| **SIMD + Multi-threaded** | **745.0**         | **\~14.81×** |
