include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================
program = main

obj = main.o force_kernel.o

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) -Xcuda-ptxas -v -v --save-temps $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: ../mcmd-cuda/main.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

force_kernel.o: force_kernel.cpp
	$(CC) -Xcuda-ptxas -v -v --save-temps $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)
	rm -rf ../mcmd-cuda/dataset/*.pdb
	rm -rf ../mcmd-cuda/dataset/*.xyz
	rm -rf ../mcmd-cuda/dataset/*.bak
	rm -rf ../mcmd-cuda/dataset/*.dat

run: $(program)
	cd ../mcmd-cuda/dataset && ../../mcmd-sycl/$(program) mcmd.inp && cd ../../mcmd-sycl
