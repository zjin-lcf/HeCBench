# wordcount-cuda/CMakeLists.txt

set(MAIN_DIR "${CMAKE_CURRENT_LIST_DIR}/../wordcount-sycl")

add_hecbench_benchmark(
    NAME wordcount
    MODEL cuda
    SOURCES ${MAIN_DIR}/main.cpp wc.cu
    CATEGORIES algorithms
)
