include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o bwt.o

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: ../bwt-cuda/main.cpp ../bwt-cuda/bwt.hpp
	$(CC) $(CFLAGS) -c $< -o $@

bwt.o: bwt.cpp ../bwt-cuda/bwt.hpp
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 10000000

