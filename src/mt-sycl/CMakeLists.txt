# mt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mt
    MODEL sycl
    SOURCES MT.cpp MT_gold.cpp genmtrand.cpp
    CATEGORIES algorithms
)
