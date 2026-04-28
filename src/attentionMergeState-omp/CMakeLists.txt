# attentionMergeState-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../attentionMergeState-sycl")

add_hecbench_benchmark(
    NAME attentionMergeState
    MODEL omp
    SOURCES main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES ml
)
