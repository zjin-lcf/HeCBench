#===============================================================================
# User Options
#===============================================================================
# Compiler can be set below, or via environment variable
CC        = mpiicpc
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu

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
CFLAGS := -std=c++14 -Wall -I../include -I${MKLROOT}/include -DMKL_ILP64 

# Linker Flags
LDFLAGS = ${MKLROOT}/lib/intel64/libmkl_sycl.a \
          -fsycl-device-code-split=per_kernel \
          -Wl,-export-dynamic -Wl,--start-group \
          ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
          ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a \
          ${MKLROOT}/lib/intel64/libmkl_core.a \
          -Wl,--end-group -ltbb -lsycl -lOpenCL -lpthread -lm -ldl 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-cxx=dpcpp -DUSE_GPU
else
  CFLAGS +=-cxx=dpcpp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(LD) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	mpirun -n 1 ./$(program) --num_vector 16 --num_field 32 --num_iterations 1
	#mpirun -n 1 ./$(program) --num_vector 4000 --num_field 90000 --num_iterations 1

