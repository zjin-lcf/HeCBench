# atomicIntrinsics-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicIntrinsics
    MODEL cuda
    SOURCES main-um.cu main.cu
    CATEGORIES algorithms
)
