#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu

#===============================================================================
# Program name & source code list
#===============================================================================

program = vj-cpu vj-gpu

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall -I../include

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

HEADERS := image.h haar.h stdio-wrapper.h

all: build

build: $(program)

image.o: image.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

stdio-wrapper.o: stdio-wrapper.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

main.o: main.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

haar-gpu.o: haar.cpp $(HEADERS)
	$(CC) $(CFLAGS) -DSYCL -o $@ -c $<

haar-cpu.o: haar.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

rectangles.o: rectangles.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

vj-cpu: main.o haar-cpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

vj-gpu: main.o haar-gpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

clean:
	rm -f $(program) *.o Output-cpu.pgm Output-gpu.pgm

run: build
	./vj-cpu ../face-cuda/Face.pgm Output-cpu.pgm
	./vj-gpu ../face-cuda/Face.pgm Output-gpu.pgm

