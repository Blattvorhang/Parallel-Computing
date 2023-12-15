#include <omp.h>
#include <thread>
#include "common.h"

extern Mode mode;

float sumSpeedUp(const float data[], const int len) {
    double sum_value = 0;
    #pragma omp parallel for num_threads(MAX_THREADS) reduction(+:sum_value)
    for (int i = 0; i < len; i++) {
        sum_value += ACCESS(data[i]);
    }
    return float(sum_value);
}


inline float max(float a, float b) {
    return a > b ? a : b;
}


inline int min(int a, int b) {
    return a < b ? a : b;
}


float maxSpeedUp(const float data[], const int len) {
    if (len <= 0)
        return 0;
    float max_value = ACCESS(data[0]);
    #pragma omp parallel for num_threads(MAX_THREADS) reduction(max:max_value)
    for (int i = 1; i < len; i++) {
        if (ACCESS(data[i]) > max_value) {
            max_value = ACCESS(data[i]);
        }
    }
    return max_value;
}


void mergeThread(
    float result[],
    const float left[],
    const float right[],
    const int left_size,
    const int right_size,
    std::thread& left_thread,
    std::thread& right_thread
) {
    // Wait for the left and right array to be sorted
    left_thread.join();
    right_thread.join();
    merge(result, left, right, left_size, right_size);
}


void sortSpeedUp(const float data[], const int len, float result[]) {
    // if (mode == LOCAL) {
    //     /* Complete binary tree */
    //     // The real number of running threads is less or equal to MAX_THREADS
    //     std::thread threads[2 * MAX_THREADS];
    //     int thread_data_size = len / MAX_THREADS;
    //     int level_start = MAX_THREADS;  // The start index of the current level
    //     int array_len;  // The length of the array to be sorted

    //     // Guarantee each leaf node is sorted
    //     for (int i = 0; i < MAX_THREADS; i++) {
    //         array_len = min(thread_data_size, len - i * thread_data_size);
    //         threads[level_start] = std::thread(
    //             mergeSort,
    //             data + i * thread_data_size,
    //             array_len,
    //             result + i * thread_data_size
    //         );
    //     }
    //     level_start /= 2;

    //     // Merge each level
    //     while (level_start > 0) {
    //         for (int i = level_start; i < 2 * level_start; i++) {
    //             array_len = min(thread_data_size, len - (i - level_start) * thread_data_size);
    //             threads[i] = std::thread(
    //                 mergeThread,
    //                 result + (i - level_start) * thread_data_size,
    //                 result + (i - level_start) * thread_data_size,
    //                 result + (i - level_start + 1) * thread_data_size,
    //                 thread_data_size,
    //                 array_len,
    //                 std::ref(threads[2 * i]),
    //                 std::ref(threads[2 * i + 1])
    //             );
    //         }
    //         level_start /= 2;
    //     }
    // }
}