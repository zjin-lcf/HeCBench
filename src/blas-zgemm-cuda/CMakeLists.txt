# blas-zgemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-zgemm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
    TEST_ARGS 256 256 256 2
    TEST_REGEX "PASS"
    TEST_TIMEOUT 300
)
