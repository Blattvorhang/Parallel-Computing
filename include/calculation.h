float sum(const float data[], const int len);
float max(const float data[], const int len);
float sort(const float data[], const int len, float result[]);

float sumSpeedUp(const float data[], const int len);
float maxSpeedUp(const float data[], const int len);
float sortSpeedUp(const float data[], const int len, float result[]);

void merge(float arr[], const float left[], const float right[], const int leftSize, const int rightSize);
void mergeSort(float arr[], const int size);