#include <cuda_runtime.h>
#include <iostream>
#include "common.h"

__device__ void merge(float* data, float* aux, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (ACCESS(data[i]) <= ACCESS(data[j])) {
            aux[k++] = data[i++];
        }
        else {
            aux[k++] = data[j++];
        }
    }

    while (i <= mid) {
        aux[k++] = data[i++];
    }

    while (j <= right) {
        aux[k++] = data[j++];
    }

    for (i = left; i <= right; i++) {
        data[i] = aux[i];
    }
}

__global__ void mergeSortKernel(float *input, float *output, long long len, int width) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = idx * (2 * width);
    
    if (start >= len) return;  // 防止越界

    int mid = min(start + width, len);
    int end = min(start + 2 * width, len);

    // 在全局内存上执行合并操作
    // 注意：这里应该使用一个专门的函数来在全局内存上执行合并
    // 下面的代码是简化的示意，你可能需要调整它以适应你的合并逻辑
    int i = start, j = mid, k = start;
    while (i < mid && j < end) {
        if (input[i] < input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }

    while (i < mid) {
        output[k++] = input[i++];
    }

    while (j < end) {
        output[k++] = input[j++];
    }
}

void sortSpeedUpCuda(const float data[], const int len, float result[]) {
    float *device_input, *device_output, *device_temp;
    // 在 GPU 上为原始数据和辅助数组分配内存
    cudaMalloc((void**)&device_input, len * sizeof(float));
    cudaMalloc((void**)&device_output, len * sizeof(float));

    // 将原始数据复制到 GPU
    cudaMemcpy(device_input, data, len * sizeof(float), cudaMemcpyHostToDevice);
    long long len_long = len;
    // 每次迭代中处理的数组部分的大小
    int width = 1;
    int blockSize = 1024;  // 线程块大小
    while (width < len) {
        int numBlocks = (len + 2 * width - 1) / (2 * width);

        mergeSortKernel<<<numBlocks, blockSize>>>(device_input, device_output, len_long, width);

        cudaDeviceSynchronize();

        // 交换输入和输出数组的指针
        device_temp = device_input;
        device_input = device_output;
        device_output = device_temp;
        width *= 2;
    }

    // 将排序后的数据复制回主机
    cudaMemcpy(result, device_input, len * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 上的内存
    cudaFree(device_input);
    cudaFree(device_output);
}
