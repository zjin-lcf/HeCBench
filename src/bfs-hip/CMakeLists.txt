# bfs-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bfs
    MODEL hip
    SOURCES bfs.cu
    CATEGORIES graph algorithms
)
