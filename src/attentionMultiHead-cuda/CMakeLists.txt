# attentionMultiHead-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMultiHead
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
