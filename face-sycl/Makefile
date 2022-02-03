include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = vj-cpu vj-gpu

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

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
