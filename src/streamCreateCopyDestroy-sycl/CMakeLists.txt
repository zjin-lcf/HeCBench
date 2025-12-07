# streamCreateCopyDestroy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamCreateCopyDestroy
    MODEL sycl
    SOURCES main-dpct.cpp main.cpp
    CATEGORIES algorithms
)
