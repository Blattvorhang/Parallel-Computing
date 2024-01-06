#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include "access.cuh"
__global__ void countSort(float* data, float* output, int len, int exp) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < len) {
        int count[2] = {0};  // 仅有两个桶：0 或 1

        // 计算每位上的0和1的数量
        for (int j = 0; j < len; j++) {
            int bit = (reinterpret_cast<int*>(&data[j])[0] >> exp) & 1;
            count[bit]++;
        }

        __syncthreads();

        // 计算累积和
        int total = 0;
        for (int j = 0; j <= (reinterpret_cast<int*>(&data[i])[0] >> exp) & 1; j++) {
            total += count[j];
        }

        __syncthreads();

        // 排序
        output[total - 1] = data[i];

        __syncthreads();

        // 将排序后的数据写回原数组
        if (i < len) {
            data[i] = output[i];
        }
    }
}

void sortSpeedUpCuda(const float data[], const int len, float result[]) {
    applyLogSqrt(data,len); //模拟任务负担
    float *dev_data, *dev_output;
    
    cudaMalloc((void**)&dev_data, len * sizeof(float));
    cudaMalloc((void**)&dev_output, len * sizeof(float));  // 为输出数组分配内存
    cudaMemcpy(dev_data, data, len * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;  // 选择一个合适的块大小
    int numBlocks = (len + blockSize - 1) / blockSize;
    for (int exp = 0; exp < 32; exp++) {
        countSort<<<numBlocks, blockSize>>>(dev_data, dev_output, len, exp);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(result, dev_output, len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
    cudaFree(dev_output);  // 释放输出数组的内存
}
