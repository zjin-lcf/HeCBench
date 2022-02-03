include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = sobol.cpp sobol_gold.cpp sobol_gpu.cpp sobol_primitives.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

sobol.o:sobol.cpp

sobol_gold.o:sobol_gold.cpp

sobol_gpu.o:sobol_gpu.cpp

sobol_primitives.o:sobol_primitives.cpp

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 1000000 1000

