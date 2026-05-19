# attentionMultiHeadKVCache-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMultiHeadKVCache
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
