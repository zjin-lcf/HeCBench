# boxfilter-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME boxfilter
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
