# index-shuffle-cuda/CMakeLists.txt

add_hecbench_benchmark(
  NAME index-shuffle
  MODEL cuda
  SOURCES main.cu
  CATEGORIES algorithms
)
