#include <iostream>
#include <ctime>
#include <algorithm>
#include "original.h"
#include "speedup.h"
#include "common.h"

extern Mode mode;

/**
 * @brief Merge two sorted arrays into one.
 * @param result The array to merge into.
 * @param left The left array to merge.
 * @param right The right array to merge.
 * @param left_size The size of the left array.
 * @param right_size The size of the right array.
 */
void merge(float result[], const float left[], const float right[], const int left_size, const int right_size) {
    int i = 0, j = 0, k = 0;

    while (i < left_size && j < right_size) {
        if (ACCESS(left[i]) < ACCESS(right[j]))
            result[k++] = left[i++];
        else
            result[k++] = right[j++];
    }

    while (i < left_size)
        result[k++] = left[i++];

    while (j < right_size)
        result[k++] = right[j++];
}


/**
 * @brief Merge two sorted arrays into one.
 * @param arr The array to merge into.
 * @param left_size The size of the left array.
 * @param right_size The size of the right array.
 */
void merge(float arr[], const int left_size, const int right_size) {
    int i = 0, j = 0, k = 0;
    float *left = arr, *right = arr + left_size;
    const int len = left_size + right_size;
    float *temp = new float[len];

    while (i < left_size && j < right_size) {
        if (ACCESS(left[i]) < ACCESS(right[j]))
            temp[k++] = left[i++];
        else
            temp[k++] = right[j++];
    }

    while (i < left_size)
        temp[k++] = left[i++];
    
    while (j < right_size)
        temp[k++] = right[j++];
    
    for (i = 0; i < len; i++)
        arr[i] = temp[i];
    
    delete[] temp;
}


/**
 * @brief Sort an array using merge sort.
 * @param arr The array to sort.
 * @param size The size of the array.
 */
void mergeSort(float arr[], const int size) {
    if (size <= 1)
        return;
    const int mid = size / 2;
    mergeSort(arr, mid);
    mergeSort(arr + mid, size - mid);
    merge(arr, mid, size - mid);
}


void run_original(
    const float data[],
    const int len,
    float& sum_value,
    float& max_value,
    float result[],
    double& sum_time,
    double& max_time,
    double& sort_time
) {
    timespec start, end;

    clock_gettime(CLOCK_REALTIME, &start);
    sum_value = sum(data, len);
    clock_gettime(CLOCK_REALTIME, &end);
    sum_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_REALTIME, &start);
    max_value = max(data, len);
    clock_gettime(CLOCK_REALTIME, &end);
    max_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_REALTIME, &start);
    sort(data, len, result);
    clock_gettime(CLOCK_REALTIME, &end);
    sort_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
}


void run_speedup(
    const float data[],
    const int len,
    float& sum_value,
    float& max_value,
    float result[],
    double& sum_time,
    double& max_time,
    double& sort_time
) {
    // TODO: Distinguish client and server mode.
    timespec start, end;

    clock_gettime(CLOCK_REALTIME, &start);
    sum_value = sumSpeedUp(data, len);
    clock_gettime(CLOCK_REALTIME, &end);
    sum_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_REALTIME, &start);
    max_value = maxSpeedUp(data, len);
    clock_gettime(CLOCK_REALTIME, &end);
    max_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_REALTIME, &start);
    sortSpeedUp(data, len, result);
    clock_gettime(CLOCK_REALTIME, &end);
    sort_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
}