include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp shrUtils.cpp block.cpp 

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

block.o: block.cpp block.h 
	$(CC) $(CFLAGS) -c $< -o $@

main.o : main.cpp kernel.cpp shrUtils.h
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o : shrUtils.cpp shrUtils.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) data/lena_std.ppm data/lena_ref.dds
