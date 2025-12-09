# xlqc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME xlqc
    MODEL cuda
    SOURCES cuda_rys_dp.cu cuda_rys_sp.cu main.cu
    CATEGORIES algorithms
)
