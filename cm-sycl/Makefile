include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp process.cpp io.cpp 

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

io.o: ../cm-cuda/io.cpp utils.h
	$(CC) -include "utils.h" $(CFLAGS) -c $< -o $@

main.o: main.cpp utils.h
	$(CC) $(CFLAGS) -c $< -o $@

process.o: process.cpp kernels.h utils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)
