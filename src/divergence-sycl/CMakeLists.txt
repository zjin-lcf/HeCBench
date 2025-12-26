# divergence-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME divergence
    MODEL sycl
    SOURCES divergence.cpp timer/timer.cpp
    CATEGORIES algorithms
)
