#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp shmem_kernels.cpp

obj = main.o shmem_kernels.o

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := -std=c++14 -Wall -I../include

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

$(program): main.o shmem_kernels.o
	$(CC) $(CFLAGS) $^ -o $@ ${LFLAGS}

main.o: main.cpp shmem_kernels.h
	$(CC) -c $(CFLAGS) $<

shmem_kernels.o: shmem_kernels.cpp
	$(CC) -c $(CFLAGS) $<

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 1000


