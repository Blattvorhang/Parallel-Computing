#pragma once

#include <cmath>

#define ACCESS(data) log(sqrt(data))

float sum(const float data[], const int len);
float max(const float data[], const int len);
void sort(const float data[], const int len, float result[]);

float sumSpeedUp(const float data[], const int len);
float maxSpeedUp(const float data[], const int len);
void sortSpeedUp(const float data[], const int len, float result[]);