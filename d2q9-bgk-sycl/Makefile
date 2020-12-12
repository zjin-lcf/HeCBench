#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu

CheckSize ?= 128x128
FINAL_STATE_FILE = ./final_state.dat
AV_VELS_FILE = ./av_vels.dat
REF_FINAL_STATE_FILE = ../check/$(CheckSize).final_state.dat
REF_AV_VELS_FILE = ../check/$(CheckSize).av_vels.dat


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
CFLAGS := -std=c++14 -Wall -I../include

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

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) Inputs/input_4096x4096.params Obstacles/obstacles_4096x4096.dat
	make check CheckSize=4096x4096

test: $(program)
	./$(program) Inputs/input_256x256.params Obstacles/obstacles_256x256.dat
	make check CheckSize=256x256


check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) \
        --ref-final-state-file=$(REF_FINAL_STATE_FILE) \
        --av-vels-file=$(AV_VELS_FILE) \
        --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(program) av_vels.dat final_state.dat $(obj)
