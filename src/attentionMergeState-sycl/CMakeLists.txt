# attentionMergeState-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMergeState
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
