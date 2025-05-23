#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
VERIFY    = no
LAUNCHER  =

# if GPU = no, then -DSYCL has not effect
GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================
program = main

obj = main.o force_kernel.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I../mcmd-cuda -fsycl \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ \
          -DSYCL

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = 

ifeq ($(GPU),yes)
  CFLAGS += -DGPU
endif

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: ../mcmd-cuda/main.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

force_kernel.o: force_kernel.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)
	rm -rf ../mcmd-cuda/dataset/*.pdb
	rm -rf ../mcmd-cuda/dataset/*.xyz
	rm -rf ../mcmd-cuda/dataset/*.bak
	rm -rf ../mcmd-cuda/dataset/*.dat

run: $(program)
	cd ../mcmd-cuda/dataset && $(LAUNCHER) ../../mcmd-sycl/$(program) mcmd.inp && cd ../../mcmd-sycl
