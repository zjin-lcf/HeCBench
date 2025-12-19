# hausdorff-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hausdorff
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
