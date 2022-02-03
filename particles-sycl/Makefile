include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp bitonicSort.cpp particles.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp particles.h

bitonicSort.o : bitonicSort.cpp bitonicSort_kernels.cpp particles.h

particles.o : particles.cpp particles_kernels.cpp particles.h

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 100

