# bsw-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsw
    MODEL cuda
    SOURCES driver.cu kernel.cu main.cu utils.cu
    CATEGORIES algorithms
)
