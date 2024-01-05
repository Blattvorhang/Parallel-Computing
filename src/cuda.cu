#include<iostream>
#include<math.h>
#include<cuda_runtime.h>
#include"device_launch_parameters.h"
#include<fstream>

#define MAX_NUM_LISTS 256

using namespace std;
int num_lists = 128; // the number of parallel threads

__device__ void radix_sort(float* const data_0, float* const data_1, \
    int num_lists, int num_data, int tid);
__device__ void merge_list(const float* src_data, float* const dest_list, \
    int num_lists, int num_data, int tid);
__device__ void preprocess_float(float* const data, int num_lists, int num_data, int tid);
__device__ void Aeprocess_float(float* const data, int num_lists, int num_data, int tid);

__global__ void GPU_radix_sort(float* const src_data, float* const dest_data, int num_lists, int num_data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 仅在需要时进行同步
    preprocess_float(src_data, num_lists, num_data, tid);
    radix_sort(src_data, dest_data, num_lists, num_data, tid);
    merge_list(src_data, dest_data, num_lists, num_data, tid);
    Aeprocess_float(dest_data, num_lists, num_data, tid);
}

__device__ void preprocess_float(float* const src_data, int num_lists, int num_data, int tid)
{
    for (int i = tid; i < num_data; i += num_lists)
    {
        unsigned int* data_temp = (unsigned int*)(&src_data[i]);
        *data_temp = (*data_temp >> 31 & 0x1) ? ~(*data_temp) : (*data_temp) | 0x80000000;
    }
}

__device__ void Aeprocess_float(float* const data, int num_lists, int num_data, int tid)
{
    for (int i = tid; i < num_data; i += num_lists)
    {
        unsigned int* data_temp = (unsigned int*)(&data[i]);
        *data_temp = (*data_temp >> 31 & 0x1) ? (*data_temp) & 0x7fffffff : ~(*data_temp);
    }
}


__device__ void radix_sort(float* const data_0, float* const data_1, \
    int num_lists, int num_data, int tid)
{
    for (int bit = 0; bit < 32; bit++)
    {
        int bit_mask = (1 << bit);
        int count_0 = 0;
        int count_1 = 0;
        for (int i = tid; i < num_data; i += num_lists)
        {
            unsigned int* temp = (unsigned int*)&data_0[i];
            if (*temp & bit_mask)
            {
                data_1[tid + count_1 * num_lists] = data_0[i]; //bug 在这里 等于时会做强制类型转化
                count_1 += 1;
            }
            else {
                data_0[tid + count_0 * num_lists] = data_0[i];
                count_0 += 1;
            }
        }
        for (int j = 0; j < count_1; j++)
        {
            data_0[tid + count_0 * num_lists + j * num_lists] = data_1[tid + j * num_lists];
        }
    }
}

__device__ void merge_list(const float* src_data, float* const dest_list, \
    int num_lists, int num_data, int tid)
{
    int num_per_list = ceil((float)num_data / num_lists);
    __shared__ int list_index[MAX_NUM_LISTS];
    __shared__ float record_val[MAX_NUM_LISTS];
    __shared__ int record_tid[MAX_NUM_LISTS];
    list_index[tid] = 0;
    record_val[tid] = 0;
    record_tid[tid] = tid;
    __syncthreads();
    for (int i = 0; i < num_data; i++)
    {
        record_val[tid] = 0;
        record_tid[tid] = tid; // bug2 每次都要进行初始化
        if (list_index[tid] < num_per_list)
        {
            int src_index = tid + list_index[tid] * num_lists;
            if (src_index < num_data)
            {
                record_val[tid] = src_data[src_index];
            }
            else {
                unsigned int* temp = (unsigned int*)&record_val[tid];
                *temp = 0xffffffff;
            }
        }
        else {
            unsigned int* temp = (unsigned int*)&record_val[tid];
            *temp = 0xffffffff;
        }
        __syncthreads();
        int tid_max = num_lists >> 1;
        while (tid_max != 0)
        {
            if (tid < tid_max)
            {
                unsigned int* temp1 = (unsigned int*)&record_val[tid];
                unsigned int* temp2 = (unsigned int*)&record_val[tid + tid_max];
                if (*temp2 < *temp1)
                {
                    record_val[tid] = record_val[tid + tid_max];
                    record_tid[tid] = record_tid[tid + tid_max];
                }
            }
            tid_max = tid_max >> 1;
            __syncthreads();
        }
        if (tid == 0)
        {
            list_index[record_tid[0]]++;
            dest_list[i] = record_val[0];
        }
        __syncthreads();
    }
}

void sortSpeedUpCuda(const float data[], const int len, float result[])
{
    float* src_data, * dest_data;
    cudaMalloc((void**)&src_data, sizeof(float) * len);
    cudaMalloc((void**)&dest_data, sizeof(float) * len);
    cudaMemcpy(src_data, data, sizeof(float) * len, cudaMemcpyHostToDevice);  // 直接从原始数据传输

    // 调整线程块的大小和数量
    int numBlocks = 1;  // 根据需求调整
    int numThreadsPerBlock = num_lists;  // 根据需求调整
    GPU_radix_sort << <numBlocks, numThreadsPerBlock >> > (src_data, dest_data, num_lists, len);

    cudaMemcpy(result, dest_data, sizeof(float) * len, cudaMemcpyDeviceToHost);
    cudaFree(src_data);
    cudaFree(dest_data);
}