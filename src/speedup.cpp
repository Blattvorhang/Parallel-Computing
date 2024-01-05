#include <omp.h>        // OpenMP
#include <immintrin.h>  // SIMD
#include <thread>
#include <algorithm>
#include "cuda.cuh"
#include "common.h"

#define SSE 0

#if SSE
/* SSE version */
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

    /* merge the two halves alternately */
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
    const double gpu_ratio = 0.5;
    const int gpu_len = int(len * gpu_ratio);
    const int cpu_len = len - gpu_len;

    const float* gpu_data = data;
    float* gpu_result = new float[gpu_len];
    float* cpu_data = new float[cpu_len];

    // 使用 std::copy 进行高效拷贝
    std::copy(data + gpu_len, data + len, cpu_data);

    std::thread thread1(sortSpeedUpCuda, gpu_data, gpu_len, gpu_result); // 创建线程1 是用GPU进行计算
    std::thread thread2(parallelSort, cpu_data, cpu_len, log2(MAX_THREADS) - 1); // 创建线程2，是用CPU进行计算

    thread1.join();
    thread2.join();

    merge(result, gpu_result, cpu_data, gpu_len, cpu_len); //归并排序

    delete [] gpu_result;
    delete [] cpu_data;
}