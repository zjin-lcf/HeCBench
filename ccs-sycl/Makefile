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

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -t 0.9 -i ../ccs-cuda/Data_Constant_100_1_bicluster.txt -o ./Output.txt -m 50 -p 1 -g 100.0
	diff -Bb Output.txt ../ccs-cuda/ccs_constant_100_1.out
