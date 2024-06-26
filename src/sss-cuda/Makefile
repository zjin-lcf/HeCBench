#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
GSL       = /path/to/GSL
LAUNCHER  =

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Xcompiler -Wall -arch=$(ARCH) \
          -I$(GSL)/include

# Linker Flags
LDFLAGS = -L$(GSL)/lib -lgsl -lblas

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

%.o: %.cu DPmixGGM.cpp  DPmixGGM_Lists.cpp  DPmixGGM_SSSmoves.cpp  \
           kernels.cu graph.cpp  gwish.cpp  utilities.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) RES/f9_n150_p50_modes_GPU.txt

run: $(program)
	$(LAUNCHER) GSL_RNG_SEED=123 ./$(program) f9_n150_p50
