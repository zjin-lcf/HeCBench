#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
ifeq ($(VENDOR), codeplay)
CC        = compute++
else
CC        = clang++
endif

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
CFLAGS := -std=c++17 -Wall -I../include

ifeq ($(VENDOR), codeplay)
CFLAGS += -I$(HOME)/computecpp/ComputeCpp-CE-2.7.0-x86_64-linux-gnu/include/ \
          -no-serial-memop -sycl -sycl-driver
else
CFLAGS += -fsycl
endif

# Linker Flags
LDFLAGS = 

ifeq ($(VENDOR), codeplay)
LDFLAGS += -lComputeCpp
endif

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

./myocyte.out:./main.o \
              ./kernel/kernel_wrapper.o \
              ./util/file/file.o \
              ./util/num/num.o \
              ./util/timer/timer.o 
	$(CC) $(CFLAGS)	./main.o \
		./kernel/kernel_wrapper.o \
		./util/file/file.o \
		./util/num/num.o \
		./util/timer/timer.o \
		$(LDFLAGS) \
		-o myocyte.out

./main.o:./common.h ./main.h ./main.c
	$(CC) $(CFLAGS)	./main.c \
		-c \
		-o ./main.o 

./kernel/kernel_wrapper.o:./common.h \
	./kernel/kernel_wrapper.h \
	./kernel/kernel_wrapper.c \
	./kernel/solver.c \
	./kernel/embedded_fehlberg_7_8.c \
	./kernel/master.c \
	./kernel/kernel_cam.sycl \
	./kernel/kernel_ecc.sycl
	$(CC) $(CFLAGS)	./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o 

./util/file/file.o:./common.h \
	./util/file/file.h \
	./util/file/file.c
	$(CC)	./util/file/file.c \
		-c \
		-o ./util/file/file.o \
		-O3

./util/num/num.o:./common.h \
	./util/num/num.h \
	./util/num/num.c
	$(CC)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

./util/timer/timer.o:./common.h \
	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

clean:
	rm	*.o *.out \
		./kernel/*.o \
		./util/file/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		./output.txt

run: myocyte.out
	./myocyte.out -time 100
