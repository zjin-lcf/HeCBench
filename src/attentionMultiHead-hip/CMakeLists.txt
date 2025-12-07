# attentionMultiHead-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMultiHead
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
