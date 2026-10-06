#===============================================================================
# User Options
#===============================================================================

CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_90
LAUNCHER  ?=

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
source = main.cu
obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -arch=$(ARCH) -I$(MPI_ROOT)/include
LDFLAGS = -L$(MPI_ROOT)/lib -l$(MPI_LIB)

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

%.o: %.cu combine.cuh reference.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --tokens 128 --hidden 1024 --iters 2 --warmup 1

.PHONY: clean run
