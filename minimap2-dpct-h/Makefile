#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = device/device_kernel_wrapper.dp.cpp \
device/device_kernel.dp.cpp \
src/host_data_io.cpp   \
src/host_kernel.cpp   \
src/main.cpp   \
src/memory_scheduler.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall -I../include -I./include \

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

$(program): $(obj) 
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp include/datatypes.h include/device_kernel_wrapper.h \
	   include/host_data_io.h include/host_kernel.h \
	   include/kernel_common.h include/memory_scheduler.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.txt

run: $(program)
	./$(program) ../minimap2-sycl/in-1k.txt kernel-out.txt

