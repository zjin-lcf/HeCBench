# lsqt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lsqt
    MODEL cuda
    SOURCES anderson.cu charge.cu hamiltonian.cu lsqt.cu main.cu model.cu model_general.cu model_lattice.cu sigma.cu vector.cu
    CATEGORIES algorithms
)
