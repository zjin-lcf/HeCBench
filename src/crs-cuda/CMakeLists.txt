# crs-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME crs
    MODEL cuda
    SOURCES GCRSMatrix.cu galois.cu jerasure.cu kernels.cu main.cu utils.cu
    CATEGORIES algorithms
)
