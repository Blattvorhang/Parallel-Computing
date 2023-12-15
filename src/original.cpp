#include "common.h"


float sum(const float data[], const int len) {
    double sum_value = 0;  // use double to guarantee the precision
    for (int i = 0; i < len; i++)
        sum_value += ACCESS(data[i]);
    return float(sum_value);
}


float max(const float data[], const int len) {
    if (len <= 0)
        return 0;
    float max_value = ACCESS(data[0]);
    for (int i = 1; i < len; i++) {
        if (ACCESS(data[i]) > max_value) {
            max_value = ACCESS(data[i]);
        }
    }
    return max_value;
}


void sort(const float data[], const int len, float result[]) {
    for (int i = 0; i < len; i++)
        result[i] = data[i];
    mergeSort(result, len);
}