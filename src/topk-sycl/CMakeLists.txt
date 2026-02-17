# topk-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blockScan-sycl/utils")

add_hecbench_benchmark(
    NAME topk
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES algorithms
)
