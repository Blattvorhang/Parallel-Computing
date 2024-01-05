#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// 核函数：对每位进行计数排序
__global__ void countSort(float *data, int len, int exp) {
    float *output = new float[len];
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

    delete[] output;
}


void sortSpeedUpCuda(const float data[], const int len, float result[]) {
    float *dev_data;
    cudaMalloc((void**)&dev_data, len * sizeof(float));
    cudaMemcpy(dev_data, data, len * sizeof(float), cudaMemcpyHostToDevice);

    // 对每一位执行基数排序
    int blockSize = 256;  // 选择一个合适的块大小
    int numBlocks = (len + blockSize - 1) / blockSize;
    for (int exp = 0; exp < 32; exp++) {
        countSort<<<numBlocks, blockSize>>>(dev_data, len, exp);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(result, dev_data, len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
}