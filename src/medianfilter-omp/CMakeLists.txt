# medianfilter-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME medianfilter
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
