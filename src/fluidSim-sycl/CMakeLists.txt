# fluidSim-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fluidSim
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -DVERIFY
)
