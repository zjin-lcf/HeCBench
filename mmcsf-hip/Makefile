#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
BOOST     = $(HOME)/boost_1_76_0

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu kernels.cu ../mmcsf-cuda/mttkrp_cpu.cu 


obj = main.o kernels.o mttkrp_cpu.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -I$(BOOST) -std=c++11 -I../mmcsf-cuda

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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu ../mmcsf-cuda/mttkrp_cpu.h mttkrp_gpu.h util.h   
	$(CC) $(CFLAGS) -c $< -o $@

mttkrp_cpu.o : ../mmcsf-cuda/mttkrp_cpu.cu ../mmcsf-cuda/mttkrp_cpu.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) -i ../mmcsf-cuda/toy.tns -m 0 -R 32 -f 128 -w 1
  
run: $(program)
	./$(program) -i ../mmcsf-cuda/output.tns -m 0 -R 32 -f 128 -w 4

