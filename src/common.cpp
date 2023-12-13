/**
 * @brief Merge two sorted arrays into one sorted array.
 * @param arr The array to merge into.
 * @param left The left array to merge.
 * @param right The right array to merge.
 * @param leftSize The size of the left array.
 * @param rightSize The size of the right array.
 */
void merge(int *arr, int *left, int *right, int leftSize, int rightSize) {
    int i = 0, j = 0, k = 0;
    while (i < leftSize && j < rightSize) {
        if (left[i] < right[j])
            arr[k++] = left[i++];
        else
            arr[k++] = right[j++];
    }
    while (i < leftSize)
        arr[k++] = left[i++];
    while (j < rightSize)
        arr[k++] = right[j++];
}


void mergeSort(int *arr, int size) {
    if (size == 1) {
        return;
    }
    int mid = size / 2;
    int *left = new int[mid];
    int *right = new int[size - mid];
    for (int i = 0; i < mid; i++) {
        left[i] = arr[i];
    }
    for (int i = mid; i < size; i++) {
        right[i - mid] = arr[i];
    }
    mergeSort(left, mid);
    mergeSort(right, size - mid);
    merge(arr, left, right, mid, size - mid);
    delete[] left;
    delete[] right;
}