# stiff-ode-sycl/CMakeLists.txt
#
# The host reference uses OpenMP. FindSYCL picks icpx or clang++ from
# Intel LLVM, and both accept -Xarch_host -fopenmp. Each driver then links
# its own runtime (icpx: libiomp5, clang++: libomp). An extra -lomp makes
# icpx load a second runtime.

add_hecbench_benchmark(
    NAME stiff-ode
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS "SHELL:-Xarch_host -fopenmp"
    TEST_ARGS --zones 2 --iters 1 --steps 2 --warmup 0
    TEST_REGEX "stiff-ode: PASS"
    TEST_TIMEOUT 300
)
