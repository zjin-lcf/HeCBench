# softmax-fused-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax-fused
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS --expt-relaxed-constexpr --use_fast_math
)
