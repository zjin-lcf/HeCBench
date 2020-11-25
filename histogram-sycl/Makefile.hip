CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \

# Standard flags 
CFLAGS := -Wall -I../include \
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

SRC = histogram_compare_base.cpp histogram_gmem_atomics.hpp histogram_smem_atomics.hpp test_util.hpp

EXE = histogram

$(EXE):$(SRC)
	$(CC) $(CFLAGS) $< -o $(EXE)  $(LDFLAGS) 

run: $(EXE)
	./$(EXE) 


clean: 
	rm -f $(EXE) $(EXE).linkinfo result*
