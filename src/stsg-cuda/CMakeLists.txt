# stsg-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME stsg
    MODEL cuda
    SOURCES main.cu Filter.cu 
    CATEGORIES algorithms
    LINK_LIBRARIES gdal
)
