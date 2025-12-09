# pingpong-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pingpong
    MODEL sycl
    SOURCES main-ccl.cpp main-mpi.cpp
    CATEGORIES algorithms
)
