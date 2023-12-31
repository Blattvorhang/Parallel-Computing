#pragma once

#include <cmath>

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /* total number of data */
#define ACCESS(data) log(sqrt(data))
#define ACCESSF(data) logf(sqrtf(data))

#define CUDA 1

enum RunningMode {
    LOCAL,
    CLIENT,
    SERVER
};

enum ProcessingType {
    ORIGINAL,
    SPEEDUP
};


void merge(float result[], const float left[], const float right[], const int left_size, const int right_size);
void merge(float arr[], const int left_size, const int right_size);
void mergeSort(float arr[], const int size);

void run_original(const float data[], const int len, float& sum_value, float& max_value, float result[], double& sum_time, double& max_time, double &sort_time);
void run_speedup(const float data[], const int len, float& sum_value, float& max_value, float result[], double& sum_time, double& max_time, double &sort_time);
