#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
BOOST     = $(HOME)/boost_1_76_0

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp kernels.cpp mttkrp_cpu.cpp 


obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall -I../include -I$(BOOST) 

# Linker Flags
LDFLAGS = -lpthread

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp mttkrp_cpu.h mttkrp_gpu.h util.h   
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) -i toy.tns -m 0 -R 32 -f 128 -w 1
  
run: $(program)
	./$(program) -i output.tns -m 0 -R 32 -f 128 -w 4

