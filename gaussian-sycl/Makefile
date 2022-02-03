include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = gaussianElim.cpp utils.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

gaussianElim.o: gaussianElim.cpp gaussianElim.h 
utils.o: utils.cpp utils.h 

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -q -t -s 4096
