# atomicIntrinsics-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicIntrinsics
    MODEL sycl
    SOURCES main-um.cpp main.cpp
    CATEGORIES algorithms
)
