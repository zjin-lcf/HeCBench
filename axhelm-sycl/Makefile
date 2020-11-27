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

program = axhelm

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall -Ddfloat=double -Ddlong=int  -I../include

# Linker Flags
LDFLAGS = ../axhelm-cuda/BlasLapack/libBlasLapack.a -lgfortran

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
all: build

build: $(program)

main.o : main.cpp axhelmReference.cpp
	$(CC) $(CFLAGS) -c $< -o $@
	
meshBasis.o : meshBasis.cpp meshBasis.hpp meshNodesTet3D.h
	$(CC) $(CFLAGS) -c $< -o $@

BlasLapack/libBlasLapack.a: 
	cd BlasLapack && make -j8 && cd ..

$(program): main.o meshBasis.o BlasLapack/libBlasLapack.a 
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS)

clean:
	cd BlasLapack && make clean && cd ..
	rm -rf $(program) main.o meshBasis.o 


# run one- and three-dimensional kernels
run: $(program)
	./$(program) 1 8000 100
	./$(program) 3 8000 100
