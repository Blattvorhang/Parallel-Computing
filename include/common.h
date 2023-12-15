#pragma once

#include <cmath>

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /* total number of data */
#define ACCESS(data) log(sqrt(data))

enum Mode {
    LOCAL,
    CLIENT,
    SERVER
};

void merge(float arr[], const float left[], const float right[], const int left_size, const int right_size, const int copy = 1);
void mergeSort(float arr[], const int size);

void run_original(const float data[], const int len, float& sum_value, float& max_value, float result[]);
void run_speedup(const float data[], const int len, float& sum_value, float& max_value, float result[]);
