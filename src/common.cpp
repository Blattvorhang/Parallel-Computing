#include "calculation.h"

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