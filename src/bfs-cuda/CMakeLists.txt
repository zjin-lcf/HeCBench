# bfs-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../bfs-sycl")

add_hecbench_benchmark(
    NAME bfs
    MODEL cuda
    SOURCES bfs.cu
    CATEGORIES graph algorithms
    INCLUDE_DIRS ${INC_DIR}
)
