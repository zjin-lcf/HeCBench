# recursiveGaussian-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME recursiveGaussian
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
