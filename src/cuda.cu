#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cfloat>
#include "cuda.cuh"
#include "../include/common.h"
#define NThreads 8

extern RunningMode mode;

#if FILL

__device__ void swap(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}

__global__ void bitonicSortKernel(float *arr, int numElements) {
    extern __shared__ float shared_arr[];
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    shared_arr[tid] = (tid < numElements) ? arr[tid] : FLT_MAX; // Fill with FLT_MAX for padding
    __syncthreads();

    for (unsigned int i = 2; i <= numElements; i <<= 1) {
        for (unsigned int j = i >> 1; j > 0; j >>= 1) {
            unsigned int tid_comp = tid ^ j;
            if (tid_comp > tid) {
                if ((tid & i) == 0) { // Ascending
                    if (shared_arr[tid] > shared_arr[tid_comp]) {
                        swap(shared_arr[tid], shared_arr[tid_comp]);
                    }
                } else { // Descending
                    if (shared_arr[tid] < shared_arr[tid_comp]) {
                        swap(shared_arr[tid], shared_arr[tid_comp]);
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid < numElements) {
        arr[tid] = shared_arr[tid];
    }
}

void sortSpeedUpCuda(const float data[], const int len, float result[]) {
    if (mode == LOCAL) {
        // Calculate the nearest power of 2 greater than or equal to len
        int power_of_2_len = 1;
        while (power_of_2_len < len) {
            power_of_2_len <<= 1;
        }

        // Create a vector to hold the extended and padded data
        std::vector<float> extended_data(power_of_2_len, 0.0f);

        // Copy the input data into the extended_data vector
        for (int i = 0; i < len; i++) {
            extended_data[i] = data[i];
        }

        float *dev_data;
        cudaMalloc((void **)&dev_data, power_of_2_len * sizeof(float));
        cudaMemcpy(dev_data, extended_data.data(), power_of_2_len * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blocks(power_of_2_len / NThreads, 1);
        dim3 threads(NThreads, 1);

        bitonicSortKernel<<<blocks, threads, power_of_2_len * sizeof(float)>>>(dev_data, power_of_2_len);

        cudaMemcpy(result, dev_data, len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_data);
    }
}
#else
__device__ void swap(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}

__global__ void bitonicSortKernel(float *arr, int numElements) {
    extern __shared__ float shared_arr[];
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    shared_arr[tid] = arr[tid];
    __syncthreads();

    for (unsigned int i = 2; i <= numElements; i <<= 1) {
        for (unsigned int j = i >> 1; j > 0; j >>= 1) {
            unsigned int tid_comp = tid ^ j;
            if (tid_comp > tid) {
                if ((tid & i) == 0) { // Ascending
                    if (shared_arr[tid] > shared_arr[tid_comp]) {
                        swap(shared_arr[tid], shared_arr[tid_comp]);
                    }
                } else { // Descending
                    if (shared_arr[tid] < shared_arr[tid_comp]) {
                        swap(shared_arr[tid], shared_arr[tid_comp]);
                    }
                }
            }
            __syncthreads();
        }
    }
    arr[tid] = shared_arr[tid];
}

void sortSpeedUpCuda(const float data[], const int len, float result[]) {
    if (mode == LOCAL) {
        // Assuming len is already a power of two
        float *dev_data;
        cudaMalloc((void **)&dev_data, len * sizeof(float));
        cudaMemcpy(dev_data, data, len * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blocks(len / NThreads, 1);
        dim3 threads(NThreads, 1);

        bitonicSortKernel<<<blocks, threads, len * sizeof(float)>>>(dev_data, len);

        cudaMemcpy(result, dev_data, len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_data);
    }
}

#endif