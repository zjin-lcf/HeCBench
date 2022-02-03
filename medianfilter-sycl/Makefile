include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp shrUtils.cpp MedianFilterHost.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp MedianFilter.cpp shrUtils.h

MedianFilterHost.o : MedianFilterHost.cpp 

shrUtils.o : shrUtils.cpp shrUtils.h 

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) ./data/SierrasRGB.ppm
