CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ocml.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_daz_opt_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/hip.bc


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
	TEST_ON = -DTEST_CHECKSUM 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif


heartwall:	./main.o \
	./kernel/kernel.o \
	./util/avi/avilib.o \
	./util/avi/avimod.o \
	./util/file/file.o \
	./util/timer/timer.o 
	$(CC)	$(CFLAGS) ./main.o \
		./kernel/kernel.o \
		./util/avi/avilib.o \
		./util/avi/avimod.o \
		./util/file/file.o \
		./util/timer/timer.o \
		$(LDFLAGS) -o heartwall

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.h \
	./main.c
	$(CC) $(CFLAGS)	$(OUTPUT) $(KERNEL_DIM) ./main.c \
		-I./util/timer/ \
		-I./util/file/ \
		-c \
		-o ./main.o 

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel.o:./kernel/kernel.sycl ./kernel/kernel.cpp
	$(CC) $(CFLAGS) $(TEST_ON) $(KERNEL_DIM)	./kernel/kernel.cpp \
		-c \
		-o ./kernel/kernel.o 

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/avi/avilib.o:./util/avi/avilib.h \
	./util/avi/avilib.c
	$(CC)	$(CFLAGS) ./util/avi/avilib.c \
		-I./util/avi \
		-c \
		-o ./util/avi/avilib.o 

./util/avi/avimod.o:./util/avi/avilib.h \
	./util/avi/avimod.c
	$(CC)	$(CFLAGS) ./util/avi/avimod.c \
		-I./util/avi \
		-c \
		-o ./util/avi/avimod.o 

./util/file/file.o:	./util/file/file.h \
	./util/file/file.c
	$(CC)	$(CFLAGS) ./util/file/file.c \
		-I./util/file \
		-I./util/timer \
		-c \
		-o ./util/file/file.o 

./util/timer/timer.o:	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC)	$(CFLAGS) ./util/timer/timer.c \
		-I./util/timer \
		-c \
		-o ./util/timer/timer.o 

run: heartwall
	./heartwall 104

clean:
	rm	-rf *.o heartwall \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o 
