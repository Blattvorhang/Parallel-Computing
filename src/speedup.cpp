#include <omp.h>
#include <thread>
#include "common.h"

extern Mode mode;

float sumSpeedUp(const float data[], const int len) {
    double sum_value = 0;
    #pragma omp parallel for num_threads(MAX_THREADS) reduction(+:sum_value)
    for (int i = 0; i < len; i++) {
        sum_value += ACCESS(data[i]);
    }
    return float(sum_value);
}


inline float max(float a, float b) {
    return a > b ? a : b;
}


float maxSpeedUp(const float data[], const int len) {
    if (len <= 0)
        return 0;
    float max_value = ACCESS(data[0]);
    #pragma omp parallel for num_threads(MAX_THREADS) reduction(max:max_value)
    for (int i = 1; i < len; i++) {
        if (ACCESS(data[i]) > max_value) {
            max_value = ACCESS(data[i]);
        }
    }
    return max_value;
}


inline int log2(int x) {
    int result = 0;
    while (x >>= 1)
        result++;
    return result;
}


/**
 * 
 * @brief This function sorts the input array in-place. It splits the array into two halves, 
 *        and sorts each half in a separate thread. Then, it merges the two halves back together.
 * @param arr The array to be sorted.
 * @param len The length of the array.
 * @param level The level of parallelism. If level is 0 or the array length is 1, 
 *              the function falls back to regular sort.
 */
void parallelSort(float arr[], const int len, const int level) {
    if (level == 0 || len <= 1) {
        mergeSort(arr, len);
        return;
    }
    const int mid = len / 2;
    std::thread t1(parallelSort, arr, mid, level - 1);
    std::thread t2(parallelSort, arr + mid, len - mid, level - 1);
    t1.join();
    t2.join();
    merge(arr, mid, len - mid);
}


void sortSpeedUp(const float data[], const int len, float result[]) {
    if (mode == LOCAL) {
        for (int i = 0; i < len; i++)
            result[i] = data[i];
        parallelSort(result, len, log2(MAX_THREADS) - 1);
    }
}