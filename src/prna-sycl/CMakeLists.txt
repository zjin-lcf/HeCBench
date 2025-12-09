# prna-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME prna
    MODEL sycl
    SOURCES prna.cpp
    CATEGORIES algorithms
)
