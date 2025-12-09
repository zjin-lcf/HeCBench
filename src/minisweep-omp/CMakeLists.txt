# minisweep-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME minisweep
    MODEL omp
    SOURCES kernels.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
