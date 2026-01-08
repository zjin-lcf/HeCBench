CC        = /opt/nvidia/hpc_sdk/Linux_aarch64/25.11/compilers/bin/nvc++
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  = ZET_ENABLE_PROGRAM_DEBUGGING=1

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
# Add CUDA include path for cuBLAS headers
CUDA_MATH_INCLUDE = /opt/nvidia/hpc_sdk/Linux_aarch64/25.11/math_libs/13.0/targets/sbsa-linux/include
CFLAGS := $(EXTRA_CFLAGS) -Wall -I$(CUDA_MATH_INCLUDE)

# Linker Flags
# Using cuBLAS instead of Intel MKL for NVIDIA GPU
LDFLAGS = -cudalib=cublas 

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
  CFLAGS +=-mp=gpu
else
  CFLAGS +=-mp=multicore
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 79 91 83 1000
	$(LAUNCHER) ./$(program) 4096 4096 4096 100
