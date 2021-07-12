#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
CUDA_ARCH = "--cuda-gpu-arch=sm_70"
GCC_TOOLCHAIN = "/auto/software/gcc/ppc64le/gcc-9.1.0"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp euclid_dist.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DUSE_GPU -std=c++17 -Wall -I../include \
          -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
          -fsycl-unnamed-lambda \
          -Xsycl-target-backend $(CUDA_ARCH) \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Dump the costs
ifeq ($(DUMP),yes)
  CFLAGS += -DDUMP
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp dtw_stencil.cpp euclid_dist.h helper_functions.cpp  
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) ../dtw-cuda/ECG200_TEST.txt

