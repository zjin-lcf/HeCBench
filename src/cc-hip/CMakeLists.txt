# cc-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mis-cuda")

add_hecbench_benchmark(
    NAME cc
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    INCLUDE_DIRS ${INC_DIR}
)
