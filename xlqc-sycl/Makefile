#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
GSL_INC   =-I/noback/5zj/gsl-2.7/include
GSL_LIB   =-L/noback/5zj/gsl-2.7/lib -lgsl -lgslcblas 

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main


obj=basis.o scf.o main.o crys.o cints.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -I../xlqc-cuda -I../xlqc-cuda/int_lib $(GSL_INC) \
          -std=c++17 -Wall -I../include -fsycl

# Linker Flags
LDFLAGS = $(GSL_LIB) 

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
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp cuda_rys_sp.cpp cuda_rys_dp.cpp rys.h
	$(CC) -c -o $@ $< $(CFLAGS)

scf.o: ../xlqc-cuda/scf.cc
	$(CC) -c -o $@ $< $(CFLAGS)

basis.o: ../xlqc-cuda/basis.cc
	$(CC) -c -o $@ $< $(CFLAGS)

crys.o: ../xlqc-cuda/int_lib/crys.cc
	$(CC) -c -o $@ $< $(CFLAGS)

cints.o: ../xlqc-cuda/int_lib/cints.cc
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) sp
	./$(program) dp

