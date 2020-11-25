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
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = KeccakF.cpp KeccakTreeCPU.cpp KeccakTreeGPU.cpp Test.cpp main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -Wall -I../include \
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH) \
	  $(ROCM_DEVICE_LIB) 

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

$(program): KeccakTreeCPU.o KeccakTreeGPU.o Test.o main.o KeccakF.o Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

KeccakTreeCPU.o: KeccakTreeCPU.cpp KeccakTreeCPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

KeccakTreeGPU.o: KeccakTreeGPU.cpp KeccakTreeGPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

Test.o: Test.cpp KeccakTreeCPU.h KeccakTreeGPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp KeccakTreeCPU.h KeccakTreeGPU.h Test.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

KeccakF.o: KeccakF.cpp KeccakTree.h KeccakF.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)

