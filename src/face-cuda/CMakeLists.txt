# face-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME face
    MODEL cuda
    SOURCES haar.cu image.cu main.cu rectangles.cu stdio-wrapper.cu
    CATEGORIES algorithms
)
