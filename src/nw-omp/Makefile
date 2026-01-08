#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = /opt/nvidia/hpc_sdk/Linux_aarch64/25.11/compilers/bin/nvc++
OPTIMIZE  = yes
DEBUG     = no
DUMP      = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = nw

source = nw.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../nw-cuda

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

ifeq ($(DUMP),yes)
  CFLAGS += -DTRACEBACK
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp  ../nw-cuda/reference.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 16384 10
