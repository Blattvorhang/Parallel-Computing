// cuda.cuh
#ifndef CUDA_CUH
#define CUDA_CUH
#include <cuda_runtime.h>
// 核函数
__global__ void mergeSortKernel(float *input, float *output, int len, int width);

void sortSpeedUpCuda(const float data[], const int len, float result[]);

#endif //CUDA_CUH
