# bsw-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsw
    MODEL sycl
    SOURCES driver.cpp kernel.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
