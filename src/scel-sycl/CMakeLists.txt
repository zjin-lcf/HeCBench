# scel-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME scel
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
