# attentionMultiHeadKVCache-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMultiHeadKVCache
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
