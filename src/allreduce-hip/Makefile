#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
MPI_ROOT  = $(HOME)/ompi_for_gpu/ompi
LAUNCHER  = $(HOME)/ompi_for_gpu/ompi/bin/mpirun -n 2

#MPI_ROOT  = /usr/lib/x86_64-linux-gnu/openmpi
#LAUNCHER  = /usr/bin/mpirun -n 1

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu collectives.cu timer.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -I$(MPI_ROOT)/include -DOMPI_SKIP_MPICXX= \
          -Wall

# Linker Flags
LDFLAGS = -L$(MPI_ROOT)/lib -lmpi -DOMPI_SKIP_MPICXX=

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu *.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
