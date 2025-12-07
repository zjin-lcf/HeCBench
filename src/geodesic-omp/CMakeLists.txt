# geodesic-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME geodesic
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
