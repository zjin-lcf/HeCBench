# attentionMultiHead-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMultiHead
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
