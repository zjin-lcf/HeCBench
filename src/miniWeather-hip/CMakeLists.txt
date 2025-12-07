# miniWeather-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME miniWeather
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
