#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu segSLIC.cu imgTrans.cu util.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

imgTrans.cu : imgTrans.h defines.h
segSLIC.cu : segSLIC.h
util.cu : util.h defines.h

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)
