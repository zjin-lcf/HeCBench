# word2vec-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME word2vec
    MODEL sycl
    SOURCES cbow.cpp word2vec.cpp
    CATEGORIES algorithms
    LINK_LIBRARIES pthread
)
