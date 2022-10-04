#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = multimat multimat_FL multimat_F

source=compact.cpp full_matrix.cpp multimat.cpp

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../include -fsycl

# Linker Flags
LDFLAGS = 

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
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

multimat: $(source)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

multimat_F: $(source)
	$(CC) $(CFLAGS) -DFUSED $^ -o $@ $(LDFLAGS)

multimat_FL: $(source)
	$(CC) $(CFLAGS) -DFUSED -DLINKED $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(program)

run: $(program)
	./multimat_FL 2000 2000
	./multimat_F 2000 2000
	./multimat 2000 2000
	./multimat_FL 2000 2000 0.3 0.05 0.05
	./multimat_F 2000 2000 0.3 0.05 0.05
	./multimat 2000 2000 0.3 0.05 0.05
