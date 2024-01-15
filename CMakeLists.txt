# cmake_minimum_required(VERSION <specify CMake version here>)
cmake_minimum_required(VERSION 3.8)
enable_language(CUDA)
# set the project name
project(SpeedUp_Computing)

# set the C++ standard to C++ 14
set(CMAKE_CXX_STANDARD 14)

# set the build type to Release
# set(CMAKE_BUILD_TYPE "Release")

# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set the CUDA flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# find OpenMP
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
else()
    message("OpenMP not found")
endif()

# find CUDA
find_package(CUDA REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

# add the executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/original.cpp
    src/speedup.cpp
    src/common.cpp
    src/client.cpp
    src/server.cpp
    src/net.cpp
    src/cuda.cu
)

# add the include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})