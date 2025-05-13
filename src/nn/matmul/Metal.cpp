#include "nn/matmul/Metal.hpp"
#include <stdexcept>

#if defined(__APPLE__)
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/QuartzCore/QuartzCore.hpp"
#endif

namespace nn::matmul {

}