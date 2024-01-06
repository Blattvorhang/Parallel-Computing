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

__global__ void mergeSortKernel(float *input, float *output, int len, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * (2 * width);

    if (start < len) {
        int mid = min(start + width, len);
        int end = min(start + 2 * width, len);

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
}



// 主机端函数：准备数据并调用核函数
void sortSpeedUpCuda(const float data[], const int len, float result[]) {
    float* device_input, * device_output, * device_temp;

    // 在 GPU 上为原始数据和辅助数组分配内存
    cudaMalloc((void**)&device_input, len * sizeof(float));
    cudaMalloc((void**)&device_output, len * sizeof(float));

    // 将原始数据复制到 GPU
    cudaMemcpy(device_input, data, len * sizeof(float), cudaMemcpyHostToDevice);

    // 每次迭代中处理的数组部分的大小
    int width = 1;

    // 计算每个线程块的大小
    int blockSize = 256; // 可以根据需要调整

    while (width < len) {
        // 计算需要的网格大小
        int numBlocks = (len + width - 1) / width;
        numBlocks = (numBlocks + blockSize - 1) / blockSize;

        // 调用核函数
        mergeSortKernel << <numBlocks, blockSize >> > (device_input, device_output, len, width);

        // 等待 GPU 完成
        cudaDeviceSynchronize();

        // 交换输入和输出数组，为下一轮迭代做准备
        device_temp = device_input;
        device_input = device_output;
        device_output = device_temp;

        // 每次迭代后加倍处理的数组部分
        width *= 2;
    }

    // 将排序后的数据复制回主机
    cudaMemcpy(result, device_input, len * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 上的内存
    cudaFree(device_input);
    cudaFree(device_output);
}