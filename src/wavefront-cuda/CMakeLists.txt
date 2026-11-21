# wavefront-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME wavefront
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
    TEST_REGEX "PASS"
    TEST_ARGS 1048576 2
)
