#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================


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
#===============================================================================
# Targets to Build
#===============================================================================

./srad:	./main.o \
	./kernel/kernel_wrapper.o \
	./util/graphics/graphics.o \
	./util/graphics/resize.o \
	./util/timer/timer.o
	$(CC) $(CFLAGS)	./main.o \
		./kernel/kernel_wrapper.o \
		./util/graphics/graphics.o \
		./util/graphics/resize.o \
		./util/timer/timer.o \
		-lm \
		$(LDFLAGS) -o srad

./main.o:./main.c ./main.h 
	$(CC)	$(CFLAGS) $(KERNEL_DIM) ./main.c \
		-c \
		-o ./main.o \
		-O3

./kernel/kernel_wrapper.o:./kernel/kernel_wrapper.h \
	./kernel/kernel_wrapper.c \
	./kernel/kernel_compress.sycl \
	./kernel/kernel_extract.sycl \
	./kernel/kernel_prepare.sycl \
	./kernel/kernel_reduce.sycl \
	./kernel/kernel_srad.sycl \
	./kernel/kernel_srad2.sycl 
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o 

./util/graphics/graphics.o:./util/graphics/graphics.h ./util/graphics/graphics.c
	$(CC)	./util/graphics/graphics.c \
		-c \
		-o ./util/graphics/graphics.o \
		-O3

./util/graphics/resize.o:./util/graphics/resize.h ./util/graphics/resize.c
	$(CC)	./util/graphics/resize.c \
		-c \
		-o ./util/graphics/resize.o \
		-O3

./util/timer/timer.o:./util/timer/timer.h ./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

clean:
	rm	./*.o srad *.pgm\
		./kernel/*.o \
		./util/graphics/*.o \
		./util/timer/*.o

run: srad
	./srad 1000 0.5 502 458

