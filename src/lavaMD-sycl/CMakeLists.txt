# lavaMD-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lavaMD
    MODEL sycl
    SOURCES main.cpp util/num/num.cpp util/timer/timer.cpp
    CATEGORIES algorithms
)
