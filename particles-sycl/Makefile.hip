#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp bitonicSort.cpp particles.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -Wall -I../include \
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH)

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

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp particles.h
	$(CC) $(CFLAGS) -c $< -o $@

bitonicSort.o : bitonicSort.cpp bitonicSort_kernels.cpp particles.h
	$(CC) $(CFLAGS) -c $< -o $@

particles.o : particles.cpp particles_kernels.cpp particles.h
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 100

