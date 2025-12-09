# tpacf-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tpacf
    MODEL sycl
    SOURCES ACF_kernel.cpp args.cpp compute.cpp histogram_kernel.cpp main.cpp model_io.cpp
    CATEGORIES algorithms
)
