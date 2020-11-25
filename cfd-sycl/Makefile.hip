CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ocml.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_daz_opt_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc

# Standard flags
CFLAGS := -Wall -I../include \
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH) \
	  $(ROCM_DEVICE_LIB) 

# Linker flags
LDFLAGS = -lm


# Enable checksum and output file
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

ifeq ($(DUMP),yes)
  CFLAGS += -DOUTPUT
endif

SRC = euler3d.cpp
EXE = euler3d

$(EXE):$(SRC) kernel_compute_flux.sycl  \
               kernel_compute_step_factor.sycl  \
	       kernel_initialize_variables.sycl \
	       kernel_memset.sycl \
	       kernel_time_step.sycl
	$(CC) $(CFLAGS) $(KERNEL_DIM) $(SRC) -o $(EXE) $(LDFLAGS)

run: $(EXE)
	./euler3d ../data/cfd/fvcorr.domn.097K 
	#./euler3d ../data/cfd/fvcorr.domn.193K   
	#./euler3d ../data/cfd/missile.domn.0.2M  

clean: 
	rm -f $(EXE) $(EXE).linkinfo result*  density momentum density_energy
