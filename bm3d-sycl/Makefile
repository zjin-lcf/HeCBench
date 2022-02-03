include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp blockmatching.cpp filtering.cpp dct8x8.cpp      

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp bm3d.hpp indices.hpp params.hpp 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) ../bm3d-cuda/noised-bufferfly-20.png denoised.png 20 color ../bm3d-cuda/bufferfly.png
