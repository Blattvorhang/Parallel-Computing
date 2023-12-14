#include <iostream>
#include <string>
#include <ctime>
#include "calculation.h"
#include "common.h"

// data to be tested (too large to be allocated on stack)
float rawFloatData[DATANUM];
float original_result[DATANUM], speedup_result[DATANUM];


int main(int argc, char const *argv[]) {
    /* check arguments */
    if (argc < 2) {
        std::cerr << "Usage: " << " [-c/--client] or [-s/--server]" << std::endl;
        return 1;
    }

    Mode mode;
    std::string arg = argv[1];
    if (arg == "-c" || arg == "--client") {
        std::cout << "Running in client mode." << std::endl;
        mode = CLIENT;
    } else if (arg == "-s" || arg == "--server") {
        std::cout << "Running in server mode." << std::endl;
        mode = SERVER;
    } else {
        std::cerr << "Unknown option: " << arg << std::endl;
        return 1;
    }
    
    /* initialize data locally */
    for (size_t i = 0; i < DATANUM; i++)
        rawFloatData[i] = float(i + 1);
    
    timespec start, end;
    double original_time, speedup_time;

    float original_sum, speedup_sum;
    float original_max, speedup_max;

    const int TESTNUM = 1;  // 5
    /* original time test */
    clock_gettime(CLOCK_REALTIME, &start);
    for (int i = 0; i < TESTNUM; i++)
        run_original(rawFloatData, DATANUM, original_sum, original_max, original_result);
    clock_gettime(CLOCK_REALTIME, &end);
    original_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    original_time /= TESTNUM;

    std::cout << "Original time consumed: " << original_time << "s" << std::endl;
    std::cout << "Original sum: " << original_sum << std::endl;
    std::cout << "Original max: " << original_max << std::endl;

    int isSorted = 1;
    for (size_t i = 0; i < DATANUM - 1; i++)
    {
        if (ACCESS(original_result[i]) > ACCESS(original_result[i + 1]))
        {
            isSorted = 0;
            break;
        }
    }
    if (isSorted)
        std::cout << "Original result is sorted." << std::endl;
    else
        std::cout << "Original result is not sorted." << std::endl;

    return 0;
}
