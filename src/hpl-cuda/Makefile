#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = mpicc
CXX       = mpicxx
FC        = mpifort
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# HPL Build Configuration
#===============================================================================

HPL_DIR = src/hpl-2.3
CONFIG_STATUS = $(HPL_DIR)/config.status

# MPI paths
MPI_INCLUDE = /usr/lib/aarch64-linux-gnu/openmpi/include

# Configure options with MPI and CUDA
CONFIGURE_OPTS = --prefix=$(PWD)/$(HPL_DIR)
CONFIGURE_OPTS += CC=$(CC) CXX=$(CXX) FC=$(FC)
CONFIGURE_OPTS += CPPFLAGS="-I$(MPI_INCLUDE) -I/usr/local/cuda/include"
CONFIGURE_OPTS += LDFLAGS="-L$(PWD)/$(HPL_DIR)/src/cuda -L/usr/local/cuda/lib64"
CONFIGURE_OPTS += LIBS="-ldgemm -lcudart -lcuda -lcublas -lopenblas -lm"

ifeq ($(DEBUG),yes)
  CONFIGURE_OPTS += --enable-debug
endif

#===============================================================================
# Targets to Build
#===============================================================================

.PHONY: all build clean run distclean

all: build

build: $(CONFIG_STATUS)
	$(MAKE) -C $(HPL_DIR)

$(CONFIG_STATUS):
	cd $(HPL_DIR) && ./configure $(CONFIGURE_OPTS)

clean:
	if [ -f $(CONFIG_STATUS) ]; then $(MAKE) -C $(HPL_DIR) clean; fi

distclean:
	if [ -f $(CONFIG_STATUS) ]; then $(MAKE) -C $(HPL_DIR) distclean; fi

run: build
	@echo "Setting up HPL GPU test..."
	@$(MAKE) -C $(HPL_DIR)/src/cuda
	@cp -f datafiles/HPL_tiny_gpu.dat $(HPL_DIR)/testing/HPL.dat
	@echo "Running HPL GPU benchmark..."
	cd $(HPL_DIR)/testing && export LD_LIBRARY_PATH=../src/cuda:/usr/local/cuda/lib64:$$LD_LIBRARY_PATH && \
	$(LAUNCHER) mpirun -n 1 ./xhpl

run-cpu: build
	@echo "Setting up HPL CPU test..."
	@cp -f datafiles/HPL_small_cpu.dat $(HPL_DIR)/testing/HPL.dat
	@echo "Running HPL CPU benchmark..."
	cd $(HPL_DIR)/testing && $(LAUNCHER) mpirun -n 1 ./xhpl

run-fp32: build
	@echo "Building FP32 CUDA library..."
	@$(MAKE) -C $(HPL_DIR)/src/cuda_fp32
	@echo "Setting up HPL FP32 GPU test..."
	@cp -f datafiles/HPL_tiny_gpu.dat $(HPL_DIR)/testing/HPL.dat
	@echo "Running HPL FP32 GPU benchmark with FP32 epsilon..."
	cd $(HPL_DIR)/testing && export LD_LIBRARY_PATH=../src/cuda_fp32:/usr/local/cuda/lib64:$$LD_LIBRARY_PATH && \
	export LD_PRELOAD=../src/cuda_fp32/libdgemm_fp32.so && \
	export HPL_USE_FP32_EPSILON=1 && \
	$(LAUNCHER) mpirun -n 1 ./xhpl

test: run
