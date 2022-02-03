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

%.o: %.cpp ../seam-carving-cuda/stb*.h utils.h kernels.h kernels_wrapper.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) ../seam-carving-cuda/Broadway_tower.jpg 10 -u 
	#./$(program) ../seam-carving-cuda/Broadway_tower.jpg 10 -a
