# Parallel Computing

## Project Overview
Accelerating array computations through multi-threading, SIMD, distributed networking (synchronous and asynchronous mechanisms) between 2 PCs running on Linux, algorithmic optimizations, and CUDA, maximizing hardware resources for enhanced performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Project File Tree](#project-file-tree)
- [Project Requirements](#project-requirements)
- [Scoring Criteria](#scoring-criteria)
- [Appendix](#appendix)

## Project File Tree
```bash
.
├── CMakeLists.txt
├── README.md
├── docs
│   └── CHANGELOG.md  # Development log
├── include           # Header files
├── src               # Source files
├── report            # Project report
├── build.bash        # Build script
├── run_local.bash    # Local execution script
├── run_client.bash   # Client execution script
└── run_server.bash   # Server execution script
```

## Project Requirements
In teams of two, utilize advanced C++ techniques including acceleration (SIMD, multithreading) and communication technologies (RPC, named pipes, HTTP, sockets) to implement functions (sum, max value, sorting) on arrays of floating-point numbers. All processing is to be distributed across two collaborating computers to maximize computational power.

## Scoring Criteria
- Higher acceleration ratios result in higher scores.
- An additional 5 points for implementations on non-Windows platforms (limited to Ubuntu, Android).
- Scores will be based on the actual testing results provided. Reports must include accurate and valid time statistics; falsification will result in penalties.

## Appendix
### (1/2) Non-accelerated Version
Provide the following functions (without acceleration) along with the time taken for each on two separate computers with the same data volume:
```cpp
float sum(const float data[], const int len); // Returns the sum of the array.
float max(const float data[], const int len); // Returns the maximum value in the array.
void sort(const float data[], const int len, float result[]); // Sorts the array and stores the result.
```

### (2/2) Dual-Machine Accelerated Version
Provide the following functions with acceleration across two machines along with their execution times:
```cpp
float sumSpeedUp(const float data[], const int len); // Returns the sum of the array with acceleration.
float maxSpeedUp(const float data[], const int len); // Returns the maximum value in the array with acceleration.
void sortSpeedUp(const float data[], const int len, float result[]); // Sorts the array with acceleration.
```
Ensure to handle SIMD instructions appropriately, considering their dependency on instruction and data lengths. Use single-precision suffix `ps` and double-precision suffix `pd`.

The code framework for testing speeds:

```cpp
QueryPerformanceCounter(&start); // Start time.
YourFunction(...); // Includes task initiation, result retrieval, collection, and synthesis.
QueryPerformanceCounter(&end); // End time.
std::cout << "Time Consumed:" << (end.QuadPart - start.QuadPart) << endl;
cout << "Sum result: " << sumResult << endl;
cout << "Max value: " << maxValue << endl;
cout << "Is sorting correct? " << isSortingCorrect << endl;
```

Note: If the data volume exceeds computational capabilities on a single machine, it can be halved. Modify `#define SUBDATANUM 2000000` to `#define SUBDATANUM 1000000` for single-machine computation. Each machine in the dual setup should handle data of size `#define SUBDATANUM 1000000` to achieve the overall computation of `#define SUBDATANUM 2000000`.
