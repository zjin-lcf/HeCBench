# miniWeather-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME miniWeather
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
