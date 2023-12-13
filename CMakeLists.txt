# cmake_minimum_required(VERSION <specify CMake version here>)
cmake_minimum_required(VERSION 3.26)

# set the project name
project(SpeedUp_Computing)

# set the C++ standard to C++ 14
set(CMAKE_CXX_STANDARD 14)

# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# add the executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        src/original.cpp
        src/speedup.cpp
        src/common.cpp
)

# add the include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
