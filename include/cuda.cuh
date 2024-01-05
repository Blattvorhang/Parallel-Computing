// cuda.cuh
#ifndef CUDA_CUH
#define CUDA_CUH
#include <cuda_runtime.h>
// 核函数
__global__ void countSort(float *data, int len, int exp);

void sortSpeedUpCuda(const float data[], const int len, float result[]);

#endif //CUDA_CUH
