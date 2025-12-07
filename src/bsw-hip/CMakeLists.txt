# bsw-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsw
    MODEL hip
    SOURCES driver.cu kernel.cu main.cu utils.cu
    CATEGORIES algorithms
)
