#include <iostream>
#include <string>
#include <ctime>
#include <random>
#include <algorithm>  // only for std::shuffle
#include "original.h"
#include "speedup.h"
#include "common.h"

#define INIT_SHUFFLE 1  // define whether to shuffle the data before sorting
#define TEST_NUM 1  // number of times to test, for calculating the average time

Mode mode;

// data to be tested (too large to be allocated on stack)
static float rawFloatData[DATANUM];
static float original_result[DATANUM], speedup_result[DATANUM];


void init(float data[], const int len) {
    for (size_t i = 0; i < len; i++)
        rawFloatData[i] = float(i + 1);

#if INIT_SHUFFLE
    // Use the same seed for random number generator to ensure the same result
    std::mt19937 rng(42);
    std::shuffle(rawFloatData, rawFloatData + len, rng);
#endif
}


/**
 * @brief Test the time consumed by a function.
 * @param data The array to be calculated.
 * @param len The length of the data.
 * @param result The result of the sorted data.
 * @param func The function to be tested.
 * @param test_num The number of times to test, for calculating the average time. (default: 5)
 * @return The time consumed by the function.
 */
double timeTest(const float data[],
                const int len,
                float result[],
                void (*func)(const float[], const int, float&, float&, float[]),
                const int test_num = 5) {

    timespec start, end;
    double time_consumed;
    float sum_value, max_value;

    clock_gettime(CLOCK_REALTIME, &start);
    for (int i = 0; i < test_num; i++)
        func(data, len, sum_value, max_value, result);
    clock_gettime(CLOCK_REALTIME, &end);
    time_consumed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    time_consumed /= test_num;

    std::cout << "Time consumed: " << time_consumed << "s" << std::endl;

    /* last time result */
    std::cout << "sum: " << sum_value << std::endl;
    std::cout << "max: " << max_value << std::endl;

    int sorted_flag = 1;
    for (size_t i = 0; i < DATANUM - 1; i++)
    {
        // std::cout << result[i] << " ";
        if (ACCESS(result[i]) > ACCESS(result[i + 1]))
        {
            sorted_flag = 0;
            break;
        }
    }
    if (sorted_flag)
        std::cout << "Result is sorted." << std::endl;
    else
        std::cout << "Result is not sorted." << std::endl;

    return time_consumed;
}


int main(int argc, char const *argv[]) {
    /* check arguments */
    if (argc < 2) {
        std::cerr << "Usage: " << " [-l | --local] [-c | --client] [-s | --server]" << std::endl;
        return 1;
    }

    std::string arg = argv[1];
    if (arg == "-l" || arg == "--local") {
        std::cout << "Running in local mode." << std::endl;
        mode = LOCAL;
    } else if (arg == "-c" || arg == "--client") {
        std::cout << "Running in client mode." << std::endl;
        mode = CLIENT;
    } else if (arg == "-s" || arg == "--server") {
        std::cout << "Running in server mode." << std::endl;
        mode = SERVER;
    } else {
        std::cerr << "Unknown option: " << arg << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    /* initialize data locally */
    init(rawFloatData, DATANUM);
    std::cout << "Data initialized." << std::endl;
    
    double original_time, speedup_time;
    double speedup_ratio;

    /* original time test */
    std::cout << "--- Original version ---" << std::endl;
    original_time = timeTest(rawFloatData, DATANUM, original_result, run_original, TEST_NUM);
    std::cout << std::endl;

    /* speedup time test */
    std::cout << "--- Speedup version ---" << std::endl;
    speedup_time = timeTest(rawFloatData, DATANUM, speedup_result, run_speedup, TEST_NUM);
    std::cout << std::endl;

    /* speedup ratio */
    speedup_ratio = original_time / speedup_time;
    std::cout << "Speedup ratio: " << speedup_ratio << std::endl;

    return 0;
}
