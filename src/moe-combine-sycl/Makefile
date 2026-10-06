#===============================================================================
# User Options
#===============================================================================

CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_90
HIP       = no
HIP_ARCH  ?= gfx90a

ifneq ($(wildcard /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include/mpi.h),)
  MPI_ROOT ?= /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
  MPI_LIB  ?= mpi_cray
else
  MPI_ROOT ?= /usr/lib/x86_64-linux-gnu/openmpi
  MPI_LIB  ?= mpi
endif

#===============================================================================
# Program name & source code list
#===============================================================================

program = main
source = main.cpp
obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -I$(MPI_ROOT)/include -I../moe-combine-cuda
LDFLAGS = -L$(MPI_ROOT)/lib -l$(MPI_LIB)

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
            -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../moe-combine-cuda/reference.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --tokens 128 --hidden 1024 --iters 2 --warmup 1

.PHONY: clean run
