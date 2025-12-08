# tgvnn-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tgvnn
    MODEL cuda
    SOURCES
        src/main.cu
        src/tgv.cu
        src/ra.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES CUDA::cufft CUDA::cublas CUDA::cusolver
    CATEGORIES imaging algorithms
)
