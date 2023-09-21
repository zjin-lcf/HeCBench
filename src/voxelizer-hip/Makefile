#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
GPU       = yes

#===============================================================================
# Program name & source code list
#===============================================================================
# path to where GLM is installed
GLM_PATH=$(HOME)/glm

# path to where Trimesh is installed
Trimesh_PATH=$(HOME)/trimesh2

program = main

source = main.cu voxelize.cu voxelize_solid.cu util_io.cpp cpu_voxelizer.cpp

obj = main.o voxelize.o voxelize_solid.o util_io.o cpu_voxelizer.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -fopenmp \
          -I$(Trimesh_PATH)/include \
          -I$(GLM_PATH)/include

# Linker Flags
LDFLAGS = $(Trimesh_PATH)/lib.Linux64/libtrimesh.a \
          $(Trimesh_PATH)/lib.Linux64/libgluit.a

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# The example supports CPU-only mode
ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp Makefile
	$(CC) -x hip $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *binvox*

run: $(program)
	./$(program) -s 256 -f ../voxelizer-cuda/xyzrgb_statuette.ply
