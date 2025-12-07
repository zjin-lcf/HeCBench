# kmeans-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmeans
    MODEL cuda
    SOURCES cluster.cu getopt.cpp kmeans.cpp read_input.cpp rmse.cpp
    CATEGORIES algorithms
)
