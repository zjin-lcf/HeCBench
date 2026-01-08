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
UPATH = ../boxfilter-sycl
SPATH = ../medianfilter-cuda

program = main

obj = main.o shrUtils.o MedianFilterHost.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I$(UPATH) -I$(SPATH)

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
  CFLAGS +=-mp=multicore
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp $(UPATH)/shrUtils.h
	$(CC) $(CFLAGS) -c $< -o $@

MedianFilterHost.o : $(SPATH)/MedianFilterHost.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o : $(UPATH)/shrUtils.cpp $(UPATH)/shrUtils.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../medianfilter-sycl/data/SierrasRGB.ppm 1000
