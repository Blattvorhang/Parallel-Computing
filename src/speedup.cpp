
#include <omp.h>        // OpenMP
//#include <immintrin.h>  // SIMD
#include <chrono>
#include <iostream>
#include <thread>
#include <algorithm>
#include "cuda.cuh"
#include "common.h"

#define SSE 0
#define TIME_TEST 0
#define RADIX 1 //更快一些

#if SSE
/* SSE version */
// 快速对数近似
__m128 fast_log2(__m128 x) {
    __m128i ix = _mm_castps_si128(x);
    __m128i exp = _mm_srli_epi32(ix, 23);
    exp = _mm_sub_epi32(exp, _mm_set1_epi32(127));

    __m128 frac = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x007FFFFF)));
    frac = _mm_or_ps(frac, _mm_castsi128_ps(_mm_set1_epi32(0x3f800000)));

    // 对 frac 使用近似多项式（例如，一个简单的线性逼近）
    __m128 log2_frac = _mm_sub_ps(frac, _mm_set1_ps(1.0f));

    __m128 log2 = _mm_cvtepi32_ps(exp);
    log2 = _mm_add_ps(log2, log2_frac);

    return log2;
}

// 计算自然对数
__m128 _mm_log_ps(__m128 x) {
    __m128 log2_x = fast_log2(x);
    return _mm_mul_ps(log2_x, _mm_set1_ps(0.693147180559945309417232121458)); // 乘以 ln(2)
}

float sumSpeedUp(const float data[], const int len) {
    __m128 sum_value_sse = _mm_setzero_ps();
    float sum_value[4] = {0};
    float access_value[4];
    int i;

    //#pragma omp parallel for num_threads(MAX_THREADS) reduction(_mm512_add_ps:sum_value_sse)
    for (i = 0; i < len - 3; i += 4) {
        // for (int j = 0; j < 4; j++) {
        //     access_value[j] = ACCESS(data[i + j]);
        // }
        // __m128 data_sse = _mm_loadu_ps(access_value);
        __m128 data_sse = _mm_loadu_ps(data + i);
        __m128 access_sse = _mm_log_ps(_mm_sqrt_ps(data_sse));
        sum_value_sse = _mm_add_ps(sum_value_sse, data_sse);
    }
    
    _mm_storeu_ps(sum_value, sum_value_sse);

    for (; i < len; i++) {
        sum_value[0] += ACCESS(data[i]);
    }

    float final_sum = 0;
    for (int i = 0; i < 4; i++) {
        final_sum += sum_value[i];
    }
    return final_sum;
}


#else
/**
 * @brief This function calculates the sum of the input array in parallel using OpenMP.
 *        It uses Kahan summation algorithm to reduce the error.
 * @param data The array to be calculated.
 * @param len The length of the data.
 * @return The sum of the array.
 */
float sumSpeedUp(const float data[], const int len) {
    float sum_value = 0.0f;
    float c = 0.0f; // A running compensation for lost low-order bits.
    #pragma omp parallel for reduction(+:sum_value, c)
    for (int i = 0; i < len; i++) {
        /* Kahan summation algorithm */
        /* https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
        float y = ACCESSF(data[i]) - c;
        float t = sum_value + y; // Low-order digits of y are lost.
        c = (t - sum_value) - y; // -(low part of y)
        sum_value = t;
    }  // Next time around, the lost low part will be added to y in a fresh attempt.
    sum_value -= c; // Correction after reduction of OpenMP.
    return sum_value;
}
#endif

// 获取浮点数的二进制表示
inline unsigned int floatToUInt(float f) {
    return *reinterpret_cast<unsigned int*>(&f);
}

// 将无符号整数的二进制表示转回浮点数
inline float uintToFloat(unsigned int ui) {
    return *reinterpret_cast<float*>(&ui);
}

// 用于基数排序的单个位的排序
void countingSortByByte(unsigned int* input, unsigned int* output, int len, int byte) {
    unsigned int count[256] = {0};

    // 计算每个字节的频率
    for (int i = 0; i < len; ++i) {
        count[(input[i] >> (byte * 8)) & 0xFF]++;
    }

    // 计算累积和
    for (int i = 1; i < 256; ++i) {
        count[i] += count[i - 1];
    }

    // 根据字节值对元素进行排序
    for (int i = len - 1; i >= 0; --i) {
        output[--count[(input[i] >> (byte * 8)) & 0xFF]] = input[i];
    }
}

void radixSort(float* data, int len) {
    unsigned int* intData = new unsigned int[len];
    unsigned int* temp = new unsigned int[len];

    // 转换为无符号整数
    for (int i = 0; i < len; ++i) {
        ACCESS(data[i]); //模拟负载
        intData[i] = floatToUInt(data[i]);
    }

    // 对每个字节进行排序
    for (int byte = 0; byte < 4; ++byte) {
        countingSortByByte(intData, temp, len, byte);
        std::swap(intData, temp);
    }

    // 如果最高位是1（负数），将这些数移到数组的前面
    int numNegatives = 0;
    for (int i = 0; i < len; ++i) {
        if (intData[i] & 0x80000000) {
            numNegatives++;
        } else {
            break;
        }
    }
    std::rotate(intData, intData + len - numNegatives, intData + len);

    // 转换回浮点数
    for (int i = 0; i < len; ++i) {
        data[i] = uintToFloat(intData[i]);
    }

    delete[] intData;
    delete[] temp;
}

void radixSortSpeedUp(float data[],const int len) {

    radixSort(data, len);
}

void mergeomp(float result[], const float left[], const float right[], const int left_size, const int right_size) {
    int total_size = left_size + right_size;

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < left_size; ++i) {
            result[i] = left[i];
        }

        #pragma omp for nowait
        for (int j = 0; j < right_size; ++j) {
            result[left_size + j] = right[j];
        }
    }

    // 合并排序后的部分
    std::inplace_merge(result, result + left_size, result + total_size);
}

float maxSpeedUp(const float data[], const int len) {
    if (len <= 0)
        return 0;
    double max_value = ACCESS(data[0]);
    #pragma omp parallel for reduction(max:max_value)
    for (int i = 1; i < len; i++) {
        double temp = ACCESS(data[i]);
        if (temp > max_value) {
            max_value = temp;
        }
    }
    return float(max_value);
}


/**
 * @brief This is an auxiliary function for parallelSort. It sorts the input array in-place.
 *        the result is stored alternately in arr1 and arr2 for faster merging.
 * @param arr1 The array to be sorted.
 * @param arr2 The array to be sorted.
 * @param len The length of the array.
 * @param level The level of parallelism. If level is 0, the function falls back to regular sort.
 */
void parallelSortAux(float arr1[], float arr2[], const int len, const int level) {
    if (level == 0) {
        mergeSort(arr1, len);
        return;
    }

    const int mid = len / 2;
    std::thread t1(parallelSortAux, arr1, arr2, mid, level - 1);
    std::thread t2(parallelSortAux, arr1 + mid, arr2 + mid, len - mid, level - 1);

    /* wait for the two arrays to be sorted */
    t1.join();
    t2.join();
    
    if (level % 2)
        merge(arr2, arr1, arr1 + mid, mid, len - mid);
    else
        merge(arr1, arr2, arr2 + mid, mid, len - mid);
}


/**
 * @brief This function sorts the input array in-place. It splits the array into two halves, 
 *        and sorts each half in a separate thread. Then, it merges the two halves back together.
 * @param arr The array to be sorted.
 * @param len The length of the array.
 * @param level The level of parallelism. If level is 0, the function falls back to regular sort.
 */
void parallelSort(float arr[], const int len, const int level) {
    float *result = arr;
    float *temp = new float[len];
    parallelSortAux(arr, temp, len, level);
    if (level % 2) {
        for (int i = 0; i < len; i++)
            arr[i] = temp[i];
    }
    delete[] temp;
}


inline int log2(int x) {
    int result = 0;
    while (x >>= 1)
        result++;
    return result;
}

void sortSpeedUp(const float data[], const int len, float result[]) {
    const double gpu_ratio = 0.23;
    const int gpu_len = int(len * gpu_ratio);
    const int cpu_len = len - gpu_len;

    const float* gpu_data = data;
    float* gpu_result = new float[gpu_len];
    float* cpu_data = new float[cpu_len];

    //使用 std::copy 进行高效拷贝
    std::copy(data + gpu_len, data + len, cpu_data);

#if !TIME_TEST
    std::thread thread1(sortSpeedUpCuda, gpu_data, gpu_len, gpu_result); 

#if RADIX
    std::thread thread2(radixSortSpeedUp, cpu_data, cpu_len); 
#else
    std::thread thread2(parallelSort, cpu_data, cpu_len, log2(MAX_THREADS) - 1);
#endif
    thread1.join();
    thread2.join();
#else
    // 时间测试
    // 开始测量 GPU 线程的时间
    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::thread thread1(sortSpeedUpCuda, gpu_data, gpu_len, gpu_result); 
    thread1.join();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // 开始测量 CPU 线程的时间
    auto start_cpu = std::chrono::high_resolution_clock::now();
#if RADIX
    std::thread thread2(radixSortSpeedUp, cpu_data, cpu_len); 
#else
    std::thread thread2(parallelSort, cpu_data, cpu_len, log2(MAX_THREADS) - 1);
#endif
    thread2.join();
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // 计算并打印运行时间
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    std::cout << "GPU time: " << gpu_time.count() << " ms\n";
    std::cout << "CPU time: " << cpu_time.count() << " ms\n";
#endif
    mergeomp(result, gpu_result, cpu_data, gpu_len, cpu_len); //归并排序

    delete [] gpu_result;
    delete [] cpu_data;
}