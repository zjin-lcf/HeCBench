include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = hotspot.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

$(obj): hotspot.cpp hotspot.h kernel_hotspot.sycl
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.out

run: $(program)
	./$(program) 512 2 200 ../data/hotspot/temp_512 ../data/hotspot/power_512 output_512.out
