CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no
DUMP      = no
DEVICE    = gpu

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = backprop.cpp  imagenet.cpp facetrain.cpp  main.cpp

obj = $(source:.cpp=.o)

# Standard flags
CFLAGS := -std=c++14 -Wall -I../include

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DUMP),yes)
  CFLAGS += -DOUTPUT
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif


$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp backprop.h bpnn_adjust_weights.sycl bpnn_layerforward.sycl
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 65536
