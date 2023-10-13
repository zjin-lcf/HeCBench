#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpc
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = divergence


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall 

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
  CFLAGS +=-qnextgen -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
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
	$(LAUNCHER) ./$(program) input.txt 10000
