# degrid-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -DPRECISION=double
)
