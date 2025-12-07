# xlqc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME xlqc
    MODEL sycl
    SOURCES cuda_rys_dp.cpp cuda_rys_sp.cpp main.cpp
    CATEGORIES algorithms
)
