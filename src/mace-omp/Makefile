#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=
FIXTURE   ?= ../data/hydragnn-sc26/mace_pipeline_capture.macepipe

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS += -qopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp mace_pipeline.hpp mace_nlist.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# The timed workload is a synthetic batch at the published SC26 dimensions:
# 128 graphs, 2726 atoms, 45974 edges. Correctness is validated on the captured
# fixture first.
run: $(program)
	$(LAUNCHER) ./$(program) --fixture $(FIXTURE) --gemm custom \
	  --warmup 100 --repeat 100
