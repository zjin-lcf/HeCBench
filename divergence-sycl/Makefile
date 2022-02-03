include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = divergence.cpp timer/timer.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

divergence.o: divergence.cpp divergence.hpp
	$(CC) $(CFLAGS) -c $< -o $@

timer.o: timer/timer.cpp timer/timer.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) input.txt 10000
