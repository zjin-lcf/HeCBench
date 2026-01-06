# word2vec-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME word2vec
    MODEL cuda
    SOURCES cbow.cu word2vec.cu
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
