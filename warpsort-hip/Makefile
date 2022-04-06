#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp warpsort.cu

obj = main.o warpsort.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DWARP_SIZE=32 -DHALF_WARP_SIZE=16 \
          -std=c++14 -I. -Wall

# Linker Flags
LDFLAGS = 

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

main.o: main.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

warpsort.o: warpsort.cu ../warpsort-cuda/cuda/CudaUtils.cuh \
                        ../warpsort-cuda/cuda/Comparators.cuh \
                        ../warpsort-cuda/cuda/DeviceTensor.cuh \
                        ../warpsort-cuda/cuda/DeviceTensor-inl.cuh \
                        ../warpsort-cuda/cuda/NumericLimits.cuh \
                        ../warpsort-cuda/cuda/Pair.cuh \
                        ../warpsort-cuda/cuda/RegisterUtils.cuh \
                        ../warpsort-cuda/cuda/ShuffleTypes.cuh \
                        ../warpsort-cuda/cuda/WarpBitonicSort.cuh \
                        Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)

