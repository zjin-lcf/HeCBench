#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpc
OPTIMIZE  = yes
DEBUG     = no
DEVICE      = gpu

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = su3_nn_bench.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall 

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-qnextgen -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) $(ONEAPI_ROOT)/compiler/latest/linux/lib/libomp-complex-fp64.o -o $@ $(LDFLAGS)

%.o: %.cpp lattice.hpp  mat_nn_openmp.hpp  su3.hpp  Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -i 100 -l 32 -t 128 -v 3 -w 1

