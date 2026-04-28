# attentionMergeState-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMergeState
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
