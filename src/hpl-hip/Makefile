.PHONY: all clean run

all:
	$(MAKE) -C src/hpl-2.3

clean:
	$(MAKE) -C src/hpl-2.3 clean

# Encodes the manual run procedure documented in README.md:
#   - copy the small-GPU input deck over HPL.dat
#   - extend LD_LIBRARY_PATH with the in-tree HIP shim and LAPACK
#   - launch xhpl under mpirun
# LAdir is exported by the bench scripts to the workspace LAPACK build;
# fall back to the README's $(HOME)/lapack path if unset.
LAdir ?= $(HOME)/lapack/build_lapack_32

run: all
	cd src/hpl-2.3/bin/intel64 && \
	    cp ../../../../datafiles/HPL_small_gpu.dat HPL.dat && \
	    LD_LIBRARY_PATH=../../src/cuda:$(LAdir)/lib:$(LD_LIBRARY_PATH) \
	        mpirun -n 1 ./xhpl
