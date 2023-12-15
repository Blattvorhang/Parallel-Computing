#include <iostream>
#include <ctime>
//#include <algorithm>
#include "original.h"
#include "speedup.h"
#include "common.h"

#define TIME_RECORD 0  // define whether to recored the time of each part

extern Mode mode;

/**
 * @brief Merge two sorted arrays into one sorted array.
 * @param result The array to merge into.
 * @param left The left array to merge.
 * @param right The right array to merge.
 * @param leftSize The size of the left array.
 * @param rightSize The size of the right array.
 */
void merge(float result[], const float left[], const float right[], const int leftSize, const int rightSize) {
    int i = 0, j = 0, k = 0;
    while (i < leftSize && j < rightSize) {
        if (ACCESS(left[i]) < ACCESS(right[j]))
            result[k++] = left[i++];
        else
            result[k++] = right[j++];
    }
    while (i < leftSize)
        result[k++] = left[i++];
    while (j < rightSize)
        result[k++] = right[j++];
}


/**
 * @brief Sort an array using merge sort.
 * @param data The array to sort.
 * @param size The size of the array.
 * @param result The array to store the result.
 */
void mergeSort(const float data[], const int size, float result[]) {
    if (size == 1)
        return;
    const int mid = size / 2;
    const float *left = data;
    const float *right = data + mid;
    const int leftSize = mid;
    const int rightSize = size - mid;
    mergeSort(left, leftSize, result);
    mergeSort(right, rightSize, result + mid);
    merge(result, left, right, leftSize, rightSize);
}


void run_original(const float data[], const int len, float& sum_value, float& max_value, float result[]) {
#if TIME_RECORD
    timespec start, end;
    double time_consumed;

    clock_gettime(CLOCK_REALTIME, &start);
    sum_value = sum(data, len);
    clock_gettime(CLOCK_REALTIME, &end);
    time_consumed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "sum time consumed: " << time_consumed << "s" << std::endl;

    clock_gettime(CLOCK_REALTIME, &start);
    max_value = max(data, len);
    clock_gettime(CLOCK_REALTIME, &end);
    time_consumed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "max time consumed: " << time_consumed << "s" << std::endl;

    /*
    clock_gettime(CLOCK_REALTIME, &start);
    std::copy(data, data + len, result);
    std::sort(result, result + len);
    clock_gettime(CLOCK_REALTIME, &end);
    time_consumed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "std::sort time consumed: " << time_consumed << "s" << std::endl;
    */

    clock_gettime(CLOCK_REALTIME, &start);
    sort(data, len, result);
    clock_gettime(CLOCK_REALTIME, &end);
    time_consumed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "sort time consumed: " << time_consumed << "s" << std::endl;
#else
    sum_value = sum(data, len);
    max_value = max(data, len);
    // sort(data, len, result);
#endif
}


void run_speedup(const float data[], const int len, float& sum_value, float& max_value, float result[]) {
    // TODO: Distinguish client and server mode.
    sum_value = sumSpeedUp(data, len);
    max_value = maxSpeedUp(data, len);
    // sortSpeedUp(data, len, result);
}