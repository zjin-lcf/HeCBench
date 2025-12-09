# bfs-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bfs
    MODEL cuda
    SOURCES bfs.cu
    CATEGORIES graph algorithms
)
