#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC    = nvcc
OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = axhelm

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++11 -Xcompiler -Wall -Ddfloat=double -Ddlong=int 

# Linker Flags
LDFLAGS = BlasLapack/libBlasLapack.a -lgfortran

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
all: build

build: $(program)

main.o : main.cu axhelmKernel.cpp axhelmReference.cpp
	$(CC) $(CFLAGS) -c $< -o $@
	
meshBasis.o : meshBasis.cpp meshBasis.hpp
	$(CC) $(CFLAGS) -c $< -o $@

$(program): main.o meshBasis.o
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS)

clean:
	rm -rf $(program) main.o meshBasis.o

# run one- and three-dimensional kernels
run: $(program)
	./$(program) 1 8000 10
	./$(program) 3 8000 10
