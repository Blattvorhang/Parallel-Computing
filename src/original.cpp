#include "common.h"
#include "calculation.h"


float sum(const float data[], const int len) {
    float sum_value = 0;
    for (int i = 1; i < len; i++)
        sum_value += ACCESS(data[i]);
    return sum_value;
}


float max(const float data[], const int len) {
    float max_value = ACCESS(data[0]);
    for (int i = 1; i < len; i++) {
        if (ACCESS(data[i]) > max_value) {
            max_value = ACCESS(data[i]);
        }
    }
    return max_value;
}


void sort(const float data[], const int len, float result[]) {
    mergeSort(data, len, result);
}