#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

sources_cu = src/numeric.cu
sources_cpp = src/main.cpp src/preprocess.cpp src/symbolic.cpp src/Timer.cpp

obj = src/main.o src/preprocess.o src/symbolic.o src/Timer.o src/numeric.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)
CFLAGS += -I./include
CFLAGS += -I./src/nicslu/include -I./src/nicslu/util

# Linker Flags
LDFLAGS = ./src/nicslu/lib/nicslu.a ./src/nicslu/util/nicslu_util.a -lrt -lpthread -lm

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

src/%.o: src/%.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

src/%.o: src/%.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 100
