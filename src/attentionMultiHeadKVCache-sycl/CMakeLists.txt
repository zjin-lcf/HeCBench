# attentionMultiHeadKVCache-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMultiHeadKVCache
    MODEL sycl
    SOURCES main.dp.cpp
    CATEGORIES ml
)
