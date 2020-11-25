CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
VERIFY    = -DVERIFY
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
  CFLAGS += -g #-DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

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
