# bfs-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bfs
    MODEL omp
    SOURCES bfs.cpp
    CATEGORIES graph algorithms
)
