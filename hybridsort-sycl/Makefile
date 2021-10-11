#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
CHECK     = yes
DUMP      = yes
TIMING    = yes

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

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++17 -Wall -I../include -fsycl

# Linker Flags
LDFLAGS = 

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

ifeq ($(CHECK),yes)
  VERIFY =-DVERIFY
endif

ifeq ($(DUMP),yes)
  OUTPUT =-DOUTPUT
endif

ifeq ($(TIMING),yes)
  TIMER =-DTIMER
endif
#===============================================================================
# Targets to Build
#===============================================================================

hybridsort: hybridsort.o bucketsort.o mergesort.o
	$(CC) $(CFLAGS) -o hybridsort $(VERIFY) $(OUTPUT) $(TIMER) \
		hybridsort.o bucketsort.o mergesort.o $(LDFLAGS)

hybridsort.o : hybridsort.c mergesort.h bucketsort.h
	$(CC) $(CFLAGS) -o hybridsort.o -c hybridsort.c

bucketsort.o: bucketsort.c \
	kernel_bucketprefix.sycl \
	kernel_histogram.sycl \
	kernel_bucketcount.sycl \
       	kernel_bucketsort.sycl  
	$(CC) $(CFLAGS) -o bucketsort.o -c bucketsort.c 

mergesort.o: mergesort.c mergesort.h kernel_mergeSortPass.sycl
	$(CC) $(CFLAGS) -o mergesort.o -c mergesort.c 

run: hybridsort
	./hybridsort r

clean:
	rm -f  *.o hybridsort
