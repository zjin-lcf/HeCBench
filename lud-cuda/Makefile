#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = lud

source = lud.cu

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Xcompiler -Wall -I./common

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): lud.cu ./common/common.cu lud_kernels.cu ./common/common.h Makefile
	$(CC) $(CFLAGS) lud.cu ./common/common.cu -o $@ $(LDFLAGS)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -v -s 2048

