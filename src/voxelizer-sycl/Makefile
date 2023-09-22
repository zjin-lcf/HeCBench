#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no

SYCL      = yes
GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================
# path to where GLM is installed
GLM_PATH=$(HOME)/glm

# path to where Trimesh is installed
Trimesh_PATH=$(HOME)/trimesh2

program = main

source = main.cpp voxelize.cpp voxelize_solid.cpp util_io.cpp cpu_voxelizer.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -fsycl -fopenmp \
          -I$(Trimesh_PATH)/include \
          -I$(GLM_PATH)/include

# Linker Flags
LDFLAGS = $(Trimesh_PATH)/lib.Linux64/libtrimesh.a \
          $(Trimesh_PATH)/lib.Linux64/libgluit.a

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
            --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# The example supports non-SYCL CPU mode
ifeq ($(SYCL),yes)
  CFLAGS +=-DUSE_SYCL
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *binvox*

run: $(program)
	./$(program) -s 256 -f ../voxelizer-cuda/xyzrgb_statuette.ply
