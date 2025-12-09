# degrid-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
