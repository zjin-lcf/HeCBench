# zmddft-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME zmddft
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -fbracket-depth=512 -ffast-math
)
