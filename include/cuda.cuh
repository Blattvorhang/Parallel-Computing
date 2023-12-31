// cuda.cuh
#ifndef CUDA_CUH
#define CUDA_CUH

#include <cuda_runtime.h>
#define FILL 1
// 声明您在 cuda.cu 中定义的任何全局函数或核函数
__global__ void bitonicSortKernel(float *dev_data, int j, int k);

// 声明任何在 cuda.cu 中定义的辅助函数
void sortSpeedUpCuda(const float data[], const int len, float result[]);

#endif //CUDA_CUH
