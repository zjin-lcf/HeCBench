CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/hip.bc 

# Standard flags
CFLAGS := -Wall -I./common\
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH) \
	  $(ROCM_DEVICE_LIB) 

# Linker flags
LDFLAGS = -lm


# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g  
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

EXE = lud

$(EXE): lud.cpp ./common/common.c ./common/common.h \
	kernel_lud_diagonal.sycl \
	kernel_lud_internal.sycl \
       	kernel_lud_perimeter.sycl
	$(CC) $(KERNEL_DIM) $(CFLAGS) -o $(EXE) lud.cpp ./common/common.c $(LDFLAGS)

run: $(EXE)
	./$(EXE) -s 8192

clean:
	rm -f $(EXE)
