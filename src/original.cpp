#include "calculation.h"

float sum(const float data[], const int len) {
    // TODO: sum up all the data
}


float max(const float data[], const int len) {
    // TODO: find the max value
}


float sort(const float data[], const int len, float result[]) {
    for (int i = 0; i < len; i++)
        result[i] = data[i];
    mergeSort(result, len);
}