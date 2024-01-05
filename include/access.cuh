// cuda.cuh
#ifndef CUDA_CUH
#define CUDA_CUH
#include <cuda_runtime.h>
// 核函数
__global__ void computeLogSqrt(float* data, int len);

void applyLogSqrt(const float* data, int len);

#endif //CUDA_CUH
