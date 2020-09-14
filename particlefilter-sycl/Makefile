VENDOR      = intel
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu


# turning the work-group size
BLOCK_SIZE=256

# Standard flags
CFLAGS := -std=c++11 -Wall -DBLOCK_SIZE=$(BLOCK_SIZE) 

# Linker flags
LDFLAGS = -lm

# Codeplay Compiler
ifneq (,$(findstring codeplay,$(VENDOR)))
  CC = compute++
  CFLAGS += -no-serial-memop -sycl -sycl-driver
  LDFLAGS += -lComputeCpp
endif

# Intel DPC++ Compiler
ifneq (,$(findstring intel,$(VENDOR)))
  CC = dpcpp
  CFLAGS += 
  LDFLAGS += 
endif

# Enable checksum and output file
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


ALL: main

main: main.o \
	kernel_find_index.sycl  \
	kernel_likelihood.sycl  \
	kernel_normalize_weights.sycl  \
	kernel_sum.sycl
	$(CC) $(CFLAGS) main.o $(LDFLAGS)  -o main 

%.o: %.[ch]
	$(CC) $(CFLAGS) $< -c

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -c

run: main
	./main -x 128 -y 128 -z 10 -np 400000
	#./main -x 128 -y 128 -z 10 -np 1024

clean:
	rm -f *.o *~ main 
