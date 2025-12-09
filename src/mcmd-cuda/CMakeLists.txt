# mcmd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcmd
    MODEL cuda
    SOURCES averages.cpp boltzmann.cpp bonding.cpp classes.cpp commy.cpp coulombic.cpp distance.cpp force_kernel.cu fugacity.cpp histogram.cpp io.cpp kernels.cpp lj.cpp main.cpp main_out.cpp mc.cpp md.cpp md_functions.cpp mixing.cpp moves.cpp observables.cpp optimize.cpp pairs.cpp polar.cpp potential.cpp radial_dist.cpp rotatepoint.cpp sp.cpp system.cpp system_functions.cpp thole_iterative.cpp thole_iterative_omp.cpp tt.cpp usefulmath.cpp
    CATEGORIES algorithms
)
