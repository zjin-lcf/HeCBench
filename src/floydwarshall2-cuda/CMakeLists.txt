# floydwarshall2-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mis-cuda")

add_hecbench_benchmark(
    NAME floydwarshall2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES graph
    INCLUDE_DIRS ${INC_DIR}
)
