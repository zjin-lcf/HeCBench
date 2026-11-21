# wavefront-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wavefront
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
    TEST_REGEX "PASS"
    TEST_ARGS 1048576 2
)
