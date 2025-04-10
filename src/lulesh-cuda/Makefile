#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
VERIFY    = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = lulesh

source = lulesh.cu \
        lulesh-viz.cu \
        lulesh-util.cu \
        lulesh-init.cu


obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Verify kernel results
ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) 
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu lulesh.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) -i 1 -s 32 -r 11 -b 1 -c 1

run: $(program)
	$(LAUNCHER) ./$(program) -i 100 -s 128 -r 11 -b 1 -c 1

