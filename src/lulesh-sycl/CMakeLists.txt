# lulesh-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lulesh
    MODEL sycl
    SOURCES lulesh-comm.cc lulesh-init.cc lulesh-util.cc lulesh-viz.cc lulesh.cc
    CATEGORIES algorithms
)
