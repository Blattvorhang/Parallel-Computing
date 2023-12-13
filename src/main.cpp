#include <iostream>
#include "calculation.h"

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)   /* total number of data */

// data to be tested
float rawFloatData[DATANUM];
float result[DATANUM];


int main() {
    // Initialize data
    for (size_t i = 0; i < DATANUM; i++)
    {
        rawFloatData[i] = float(i + 1);
    }
    
    // test
    mergeSort(rawFloatData, DATANUM, result);
    int isSorted = 1;
    for (size_t i = 0; i < DATANUM - 1; i++)
    {
        if (ACCESS(result[i]) > ACCESS(result[i + 1]))
        {
            isSorted = 0;
            break;
        }
    }
    if (isSorted)
        std::cout << "The result is sorted." << std::endl;
    else
        std::cout << "The result is not sorted." << std::endl;

    return 0;
}
