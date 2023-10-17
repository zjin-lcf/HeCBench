#===============================================================================
# User Options
#===============================================================================
MPI_ROOT=$(HOME)/openmpi-4.1.0-install/
HIPBLAS_ROOT=/opt/rocm/hipblas
HIPRT_ROOT=/opt/rocm/hip

# Compiler can be set below, or via environment variable
CC        = hipcc
LD        = $(MPI_ROOT)/bin/mpiCC
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -I$(MPI_ROOT)/include -std=c++11 -Wall -fPIE

# Linker Flags
LDFLAGS = -L$(HIPBLAS_ROOT)/lib -lhipblas \
	  -L$(HIPRT_ROOT)/lib -lamdhip64 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(LD) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) $(MPI_ROOT)/bin/mpirun -n 4 ./$(program) --num_vector 4000 --num_field 90000 --num_iterations 10
