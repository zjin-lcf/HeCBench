CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = sgemm

source = gemm.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -Wall -I${MKLROOT}/include

# Linker Flags
LDFLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	  ${MKLROOT}/lib/intel64/libmkl_sequential.a \
	  ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group  \
	  -lOpenCL  -lpthread -ldl -lm 

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
  CFLAGS +=-DMKL_ILP64 -D__STRICT_ANSI__ -m64 \
           -fiopenmp -fopenmp-targets=spir64  -mllvm -vpo-paropt-use-interop
else
  CFLAGS +=-qopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 100
