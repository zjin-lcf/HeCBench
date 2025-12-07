# haversine-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME haversine
    MODEL omp
    SOURCES distance.cpp
    CATEGORIES algorithms
)
