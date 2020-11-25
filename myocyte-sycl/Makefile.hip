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
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_unsafe_math_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_finite_only_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_isa_version_906.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/hip.bc


# Standard flags
CFLAGS := -Wall -I../include \
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH) \
	  $(ROCM_DEVICE_LIB) 


# Linker flags
LDFLAGS = -lm


# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g -DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif


ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif
# ========================================================================================================================================================================================================200
#	EXECUTABLES (LINK OBJECTS TOGETHER INTO BINARY)
# ========================================================================================================================================================================================================200

./myocyte.out:	./main.o \
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

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./common.h \
	./main.h \
	./main.c
	$(CC) $(CFLAGS)	./main.c \
		-c \
		-o ./main.o 

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_wrapper.o:	./common.h \
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

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/file/file.o:	./common.h \
	./util/file/file.h \
	./util/file/file.c
	$(CC) $(CFLAGS)	./util/file/file.c \
		-c \
		-o ./util/file/file.o \
		-O3

./util/num/num.o:	./common.h \
	./util/num/num.h \
	./util/num/num.c
	$(CC) $(CFLAGS)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

./util/timer/timer.o:	./common.h \
	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC) $(CFLAGS)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3


run: myocyte.out
	./myocyte.out -time 100

clean:
	rm	*.o *.out \
		./kernel/*.o \
		./util/file/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		./output.txt
