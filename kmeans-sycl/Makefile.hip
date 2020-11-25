CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
INORDER   = no
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc

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

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif

ifeq ($(INORDER),yes)
  CFLAGS +=-DUSE_IOQ
endif

all: kmeans

kmeans: cluster.o getopt.o read_input.o rmse.o kmeans.o
	$(CC) $(KERNEL_DIM) $(CFLAGS) $^ -o kmeans $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $< -c

kmeans.o: kmeans.c kmeans.h

cluster.o: cluster.c kmeans.h

getopt.o: getopt.c getopt.h

read_input.o: read_input.c kmeans.h

rmse.o: rmse.c kmeans.h

clean:
	rm -f *.o *~ kmeans *.linkinfo

run: kmeans
	./kmeans -r -n 5 -m 15 -l 10 -o -i ../data/kmeans/kdd_cup


