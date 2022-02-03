include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp meshBasis.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o : main.cpp axhelmReference.cpp
	$(CC) $(CFLAGS) -c $< -o $@
	
meshBasis.o : meshBasis.cpp meshBasis.hpp meshNodesTet3D.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# run one- and three-dimensional kernels
run: $(program)
	./$(program) 1 8000 100
	./$(program) 3 8000 100
