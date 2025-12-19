# debayer-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../debayer-sycl")

add_hecbench_benchmark(
    NAME debayer
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    INCLUDE_DIRS ${INC_DIR}
)
