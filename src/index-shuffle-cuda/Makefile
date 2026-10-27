#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

%.o: %.cu kernels.h reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Production Llama 4 Scout / Maverick IndexShuffling sizes (Top-1):
#   tokens in {128, 2048, 4096, 8192}, experts in {16, 128}
run: $(program)
	$(LAUNCHER) ./$(program) 128 16 1 1000
	$(LAUNCHER) ./$(program) 128 128 1 1000
	$(LAUNCHER) ./$(program) 2048 16 1 1000
	$(LAUNCHER) ./$(program) 2048 128 1 1000
	$(LAUNCHER) ./$(program) 4096 16 1 1000
	$(LAUNCHER) ./$(program) 4096 128 1 1000
	$(LAUNCHER) ./$(program) 8192 16 1 1000
	$(LAUNCHER) ./$(program) 8192 128 1 1000
