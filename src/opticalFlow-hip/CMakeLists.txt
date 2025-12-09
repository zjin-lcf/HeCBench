# opticalFlow-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME opticalFlow
    MODEL hip
    SOURCES flowGold.cu flowHIP.cu main.cu
    CATEGORIES algorithms
)
