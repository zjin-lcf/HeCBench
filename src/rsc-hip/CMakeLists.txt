# rsc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsc
    MODEL hip
    SOURCES main.cu model_eval.cu
    CATEGORIES algorithms
)
