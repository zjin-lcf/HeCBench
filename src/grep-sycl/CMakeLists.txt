# grep-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME grep
    MODEL sycl
    SOURCES main.cpp nfautil.cpp pnfa.cpp putil.cpp regex.cpp
    CATEGORIES algorithms
)
