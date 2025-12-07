# opticalFlow-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME opticalFlow
    MODEL sycl
    SOURCES flowGold.cpp flowSYCL.cpp main.cpp
    CATEGORIES algorithms
)
