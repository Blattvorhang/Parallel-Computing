#include <iostream>
#include <string>
#include <ctime>
#include <random>
#include <algorithm>  // only for std::shuffle
#include "original.h"
#include "speedup.h"
#include "common.h"
#include "client.h"
#include "server.h"

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
    std::cout << "Shuffling data..." << std::endl;
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
double timeTest(
    const float data[],
    const int len,
    float result[],
    void (*func)(const float[], const int, float&, float&, float[], double&, double&, double&),
    double& sum_time,
    double& max_time,
    double& sort_time,
    const int test_num = 5
) {
    timespec start, end;
    double time_consumed;
    double single_sum_time, single_max_time, single_sort_time;
    double total_sum_time = 0, total_max_time = 0, total_sort_time = 0;
    float sum_value, max_value;

    for (int i = 0; i < test_num; i++) {
        func(data, len, sum_value, max_value, result, single_sum_time, single_max_time, single_sort_time);
        total_sum_time += single_sum_time;
        total_max_time += single_max_time;
        total_sort_time += single_sort_time;
    }

    sum_time = total_sum_time / test_num;
    max_time = total_max_time / test_num;
    sort_time = total_sort_time / test_num;
    time_consumed = sum_time + max_time + sort_time;
    std::cout << "Sum time consumed: " << sum_time << std::endl;
    std::cout << "Max time consumed: " << max_time << std::endl;
    std::cout << "Sort time consumed: " << sort_time << std::endl;
    std::cout << "Total time consumed: " << time_consumed << std::endl;
    std::cout << std::endl;

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
    /**
     * Usage: [-l | --local]
     *        [-c | --client <server_ip> <server_port>]
     *        [-s | --server <server_port>]
     *        [-t | --test <test_num>]
     *        [-h | --help]
     */
    if (argc < 2) {
        std::cerr << "Usage: [-l | --local] [-c | --client <server_ip> <server_port>] [-s | --server <server_port>]" << std::endl;
        return 1;
    }

    std::string arg = argv[1];
    std::string server_ip;
    int server_port;
    if (arg == "-l" || arg == "--local") {
        std::cout << "Running in local mode." << std::endl;
        mode = LOCAL;
    } else if (arg == "-c" || arg == "--client") {
        if (argc < 4) {
            std::cerr << "Client mode requires server IP and port." << std::endl;
            return 1;
        }
        server_ip = argv[2];
        server_port = std::stoi(argv[3]);
        std::cout << "Running in client mode." << std::endl;
        std::cout << "Server IP: " << server_ip << ", Server Port: " << server_port << std::endl;
        mode = CLIENT;
    } else if (arg == "-s" || arg == "--server") {
        if (argc < 3) {
            std::cerr << "Server mode requires server port." << std::endl;
            return 1;
        }
        server_port = std::stoi(argv[2]);
        std::cout << "Running in server mode." << std::endl;
        std::cout << "Server Port: " << server_port << std::endl;
        mode = SERVER;
    } else {
        std::cerr << "Unknown option: " << arg << std::endl;
        return 1;
    }
    
    /* initialize data locally */
    std::cout << "Initializing data..." << std::endl;
    init(rawFloatData, DATANUM);
    std::cout << "Data initialized." << std::endl;
    
    /* connect to server */
    if (mode == CLIENT) {
        int ret = clientConnect(server_ip.c_str(), server_port);
        if (ret == -1) {
            std::cerr << "Error connecting to server" << std::endl;
            return 1;
        }
    } else if (mode == SERVER) {
        int ret = serverConnect(server_port);
        if (ret == -1) {
            std::cerr << "Error creating server" << std::endl;
            return 1;
        }
        return 0;
    }

    double original_time, speedup_time;
    double original_sum_time, original_max_time, original_sort_time;
    double speedup_sum_time, speedup_max_time, speedup_sort_time;
    double sum_speedup_ratio, max_speedup_ratio, sort_speedup_ratio;
    double speedup_ratio;

    std::cout << "Time test begins." << std::endl << std::endl;

    /* original time test */
    std::cout << "--- Original version ---" << std::endl;
    original_time = timeTest(
        rawFloatData,
        DATANUM,
        original_result,
        run_original,
        original_sum_time,
        original_max_time,
        original_sort_time,
        TEST_NUM
    );
    std::cout << std::endl;

    /* speedup time test */
    std::cout << "--- Speedup version ---" << std::endl;
    speedup_time = timeTest(
        rawFloatData,
        DATANUM,
        speedup_result,
        run_speedup,
        speedup_sum_time,
        speedup_max_time,
        speedup_sort_time,
        TEST_NUM
    );
    std::cout << std::endl;

    /* speedup ratio */
    std::cout << "--- Speedup ratio ---" << std::endl;
    sum_speedup_ratio = original_sum_time / speedup_sum_time;
    max_speedup_ratio = original_max_time / speedup_max_time;
    sort_speedup_ratio = original_sort_time / speedup_sort_time;
    speedup_ratio = original_time / speedup_time;
    std::cout << "Sum speedup ratio: " << sum_speedup_ratio << std::endl;
    std::cout << "Max speedup ratio: " << max_speedup_ratio << std::endl;
    std::cout << "Sort speedup ratio: " << sort_speedup_ratio << std::endl;
    std::cout << "Total speedup ratio: " << speedup_ratio << std::endl;

    return 0;
}
