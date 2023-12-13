#include <iostream>

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /* total number of data */

// data to be tested
float rawFloatData[DATANUM];


int main() {
    // Initialize data
    for (size_t i = 0; i < DATANUM; i++)
    {
        rawFloatData[i] = float(i + 1);
    }
    
    // test
    float result = 0;
    for (size_t i = 0; i < DATANUM; i++)
    {
        result += rawFloatData[i];
    }
    std::cout << "result: " << result << std::endl;

    return 0;
}
