#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

c_source = main.c base.c param.c util.c
cuda_source = prna.cu 

obj = $(c_source:.c=.o) $(cuda_source:.cu=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DFLOAT -D__HIP_PLATFORM_AMD__ -Wall

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -ffast-math
endif

#===============================================================================
# Targets to Build
#===============================================================================

# generate a binary seq file
#gen: save-params.c param.c base.c util.c
#	gcc -O2 -DFLOAT -o gen-bin-seq save-params.c param.c base.c util.c -lm

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) ../prna-cuda/test.seq

run: $(program)
	./$(program) ../prna-cuda/HIV1-NL43.seq
