#pragma once

void merge(float arr[], const float left[], const float right[], const int leftSize, const int rightSize);
void mergeSort(const float data[], const int size, float result[]);

void run_original(const float data[], const int len, float& sum_value, float& max_value, float result[]);
void run_speedup(const float data[], const int len, float& sum_value, float& max_value, float result[]);