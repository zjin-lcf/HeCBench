# dp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dp
    MODEL hip
    SOURCES main.cu shrUtils.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS --hipstdpar 
                    --hipstdpar-path=/opt/rocm/include/thrust/system/hip/hipstdpar
                    -munsafe-fp-atomics
    LINK_LIBRARIES hipblas
)
