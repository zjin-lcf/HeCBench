# permutate-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME permutate
    MODEL cuda
    SOURCES gpu_permutation_testing.cu main.cpp permutation_testing.cpp statistical_test.cpp utils.cpp bzip/blocksort.cpp bzip/bzlib.cpp bzip/compress.cpp bzip/crctable.cpp bzip/decompress.cpp bzip/huffman.cpp bzip/randtable.cpp
    CATEGORIES algorithms
)
