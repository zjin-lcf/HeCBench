cmake_minimum_required(VERSION 3.21)

project(set_intersection_eval HIP CXX)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip)
find_package(hipblas)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)

# CXX options library
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)
# Adds cxxopts

# Formatting library
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        7.1.2
)
FetchContent_MakeAvailable(fmt)
# Adds fmt::fmt

# Zipf header library
FetchContent_Declare(
        zipf
        GIT_REPOSITORY https://github.com/llersch/cpp_random_distributions.git
        GIT_TAG        "6ae38ecf5dbeb1150b80ecf051cb43cb88ba697a"
)
FetchContent_MakeAvailable(zipf)

# Progress bar library
FetchContent_Declare(
        progressbar
        GIT_REPOSITORY https://github.com/prakhar1989/progress-cpp.git
        GIT_TAG        "7bfba0d22d19c41323aa35541618b6ebec9d737c"
)
FetchContent_MakeAvailable(progressbar)


# Generate dataset binary
add_executable(generate_dataset src/generate_dataset.cpp)
target_include_directories(generate_dataset PRIVATE include ${zipf_SOURCE_DIR} ${progressbar_SOURCE_DIR}/include/progresscpp/)
target_link_libraries(generate_dataset PRIVATE cxxopts fmt::fmt)

# Optimized binary search binary using matrix multiplication
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE cxxopts fmt::fmt hip::host hip::device roc::hipblas)
target_include_directories(main PRIVATE include)
target_compile_options(main PRIVATE -m64)
