#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int \
	-DMINIFE_RESTRICT=__restrict__

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX

#-----------------------------------------------------------------------
CXX       = clang++
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"
LAUNCHER       =

CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) \
           -std=c++17 -Wall -fsycl

ifeq ($(CUDA), yes)
  CPPFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
              -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
              --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(HIP), yes)
  CPPFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	      -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  CPPFLAGS += -DMINIFE_DEBUG
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

CXXFLAGS = $(CFLAGS)

ifeq ($(GPU),yes)
  CPPFLAGS +=-DUSE_GPU
endif

LIBS=

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

include make_targets
