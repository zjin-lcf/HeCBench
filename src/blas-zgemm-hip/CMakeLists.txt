# blas-zgemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-zgemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    COMPILE_OPTIONS -DHIPBLAS_V2
    LINK_LIBRARIES hipblas
    TEST_ARGS 256 256 256 2
    TEST_REGEX "PASS"
    TEST_TIMEOUT 300
)
