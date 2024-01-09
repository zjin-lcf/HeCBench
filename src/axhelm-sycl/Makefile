#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o meshBasis.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -Ddfloat=float -Ddlong=int \
          -std=c++17 -Wall -I../include -fsycl

# Linker Flags
LDFLAGS = ./BlasLapack/libBlasLapack.a -lgfortran

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
            --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) BlasLapack/libBlasLapack.a
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS)

main.o : main.cpp axhelmReference.cpp
	$(CC) $(CFLAGS) -c $< -o $@
	
meshBasis.o : meshBasis.cpp meshBasis.hpp meshNodesTet3D.h
	$(CC) $(CFLAGS) -c $< -o $@

BlasLapack/libBlasLapack.a: 
	cd BlasLapack && make -j8 && cd ..

clean:
	rm -rf $(program) $(obj)

# run one- and three-dimensional kernels
run: $(program)
	$(LAUNCHER) ./$(program) 1 8000 100
	./$(program) 3 8000 100
