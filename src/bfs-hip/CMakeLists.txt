# bfs-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../bfs-sycl")

add_hecbench_benchmark(
    NAME bfs
    MODEL hip
    SOURCES bfs.cu
    CATEGORIES graph algorithms
    INCLUDE_DIRS ${INC_DIR}
)
