// cuda.cuh
#ifndef CUDA_CUH
#define CUDA_CUH

#include <cuda_runtime.h>
#define FILL 1
__global__ void GPU_radix_sort(float* const src_data, float* const dest_data, \
    int num_lists, int num_data);

// 声明任何在 cuda.cu 中定义的辅助函数
void sortSpeedUpCuda(const float data[], const int len, float result[]);

#endif //CUDA_CUH
