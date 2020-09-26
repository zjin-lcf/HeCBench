#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
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
CFLAGS := -std=c++14 -Wall -I../include -DUSE_WORKAROUND -DMILC_COMPLEX

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
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp lattice.hpp  mat_nn_sycl.hpp su3.h su3.hpp  Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -i 100 -l 32 -t 128 -v 3 -w 1

