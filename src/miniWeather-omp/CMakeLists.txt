# miniWeather-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME miniWeather
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
