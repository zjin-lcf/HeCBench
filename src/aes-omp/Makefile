#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = /opt/nvidia/hpc_sdk/Linux_aarch64/25.11/compilers/bin/nvc++
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =


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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../include

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
  CFLAGS +=-mp=gpu
else
  # Use -mp=multicore for GCC/Clang, -mp=multicore for Intel
  ifeq ($(CC),icpx)
    CFLAGS +=-mp=multicore
  else
    CFLAGS +=-mp=multicore
  endif
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

$(obj): main.cpp aes.h ../include/SDKBitMap.h kernels.cpp utils.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# reuse the image
run: $(program)
	$(LAUNCHER) ./$(program) 100 0 ../urng-sycl/URNG_Input.bmp
	$(LAUNCHER) ./$(program) 100 1 ../urng-sycl/URNG_Input.bmp
