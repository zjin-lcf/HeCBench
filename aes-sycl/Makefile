include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

$(obj): main.cpp aes.h ../include/SDKBitMap.h kernels.cpp utils.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) $(NUM_PASSES) 0 ../urng-sycl/URNG_Input.bmp
#	./$(program) 100 1 ../urng-sycl/URNG_Input.bmp

