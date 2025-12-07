# pns-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pns
    MODEL sycl
    SOURCES main.cpp petri_kernel.cpp rand_gen.cpp
    CATEGORIES algorithms
)
