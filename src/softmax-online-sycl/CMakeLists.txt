# softmax-online-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME softmax-online
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
