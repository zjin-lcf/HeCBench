include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp kernels.cpp mttkrp_cpu.cpp 

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp mttkrp_cpu.h mttkrp_gpu.h util.h   
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) -i toy.tns -m 0 -R 32 -f 128 -w 1
  
run: $(program)
	./$(program) -i output.tns -m 0 -R 32 -f 128 -w 4

