#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/hip.bc

#===============================================================================
# Program name & source code list
#===============================================================================

program = divergence


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -Wall -I../include \
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH) \
	  $(ROCM_DEVICE_LIB) 

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

$(program): divergence.o timer.o 
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

divergence.o: divergence.cpp divergence.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

timer.o: timer/timer.cpp timer/timer.hpp
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -rf $(program) *.o

run: $(program)
	./$(program) input.txt 10000
