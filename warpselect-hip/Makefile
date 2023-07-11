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

source = main.cu DeviceUtils.cu FaissException.cu WarpSelectFloat.cu \
         WarpSelectFloat128.cu WarpSelectFloat32.cu WarpSelectFloatF2048.cu \
         WarpSelectFloatT2048.cu WarpSelectFloat1.cu WarpSelectFloat64.cu \
         WarpSelectFloatF512.cu WarpSelectFloatT512.cu WarpSelectFloat256.cu \
         WarpSelectFloatF1024.cu   WarpSelectFloatT1024.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu *.cuh
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 10000 40000 2048 100
