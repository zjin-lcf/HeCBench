# softmax-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml math
)
