include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp utils.cpp heap.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

heap.o: heap.cpp heap.h

utils.o: utils.cpp heap.h utils.h

main.o: main.cpp utils.h kernel_metropolis.h kernel_prng.h kernel_reduction.h  

clean:
	rm -rf $(program) $(obj)

dbg: $(program)
	./$(program) -l 32 11 -t 4.7 0.1 -a 1 1 1 1 -h 1.0 -z 7919

run: $(program)
	./$(program) -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919
