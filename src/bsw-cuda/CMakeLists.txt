# bsw-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsw
    MODEL cuda
    SOURCES driver.cu main.cu utils.cu
    CATEGORIES algorithms
)
