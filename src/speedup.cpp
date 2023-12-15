#include <omp.h>
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


float max(float a, float b) {
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


void sortSpeedUp(const float data[], const int len, float result[]) {
    // TODO: sort the data
}