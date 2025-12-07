# attentionMultiHead-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMultiHead
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
