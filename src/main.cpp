#include <iostream>

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /* total number of data */

// data to be tested
float rawFloatData[DATANUM];

void add(float *arr, int size) {
    // TODO: add elements in arr
}

void max(float *arr, int size) {
    // TODO: find the max element in arr
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
    int i = 0, j = 0, k = 0;
    while (i < mid && j < size - mid) {
        if (left[i] < right[j]) {
            arr[k] = left[i];
            i++;
            k++;
        } else {
            arr[k] = right[j];
            j++;
            k++;
        }
    }
    while (i < mid) {
        arr[k] = left[i];
        i++;
        k++;
    }
    while (j < size - mid) {
        arr[k] = right[j];
        j++;
        k++;
    }
    delete[] left;
    delete[] right;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
