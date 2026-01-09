# bfs-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../bfs-sycl")

add_hecbench_benchmark(
    NAME bfs
    MODEL omp
    SOURCES bfs.cpp
    CATEGORIES graph algorithms
    INCLUDE_DIRS ${INC_DIR}
)
