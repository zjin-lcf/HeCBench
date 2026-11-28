#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_80
HIP       = no
HIP_ARCH  ?= gfx90a
ONEMATH   = no
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

# path to the install of oneMath (NVIDIA and AMD targets); its cuBLAS or
# rocBLAS backend must support gemm_batch with INT8 inputs and INT32 output
# (https://github.com/uxlfoundation/oneMath/pull/803 for cuBLAS,
#  https://github.com/uxlfoundation/oneMath/pull/804 for rocBLAS)
ONEMATH_PATH  ?= /path/to/oneMath

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

ifeq ($(CUDA), yes)
  ONEMATH = yes
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
  LDFLAGS = -L$(ONEMATH_PATH)/lib -Wl,-rpath,$(ONEMATH_PATH)/lib \
            -lonemath -lonemath_blas_cublas
endif

ifeq ($(HIP), yes)
  ONEMATH = yes
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
  LDFLAGS = -L$(ONEMATH_PATH)/lib -Wl,-rpath,$(ONEMATH_PATH)/lib \
            -lonemath -lonemath_blas_rocblas
endif

ifeq ($(ONEMATH), yes)
  CFLAGS += -DUSE_ONEMATH -I$(ONEMATH_PATH)/include
else
  CFLAGS += -qmkl=parallel
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

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../blas-zgemm-cuda/ozaki2.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 256 256 256 10
	$(LAUNCHER) ./$(program) 4096 4096 4096 20
	$(LAUNCHER) ./$(program) 8192 8192 8192 10
