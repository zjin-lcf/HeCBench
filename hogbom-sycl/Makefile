include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

inc = ../hogbom-cuda

obj = main.o kernels.o reference.o timer.o

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: $(inc)/main.cpp $(inc)/utils.h $(inc)/timer.h $(inc)/reference.h $(inc)/kernels.h
	$(CC) $(CFLAGS) -c $< -o $@

reference.o: $(inc)/reference.cpp $(inc)/reference.h 
	$(CC) $(CFLAGS) -c $< -o $@

timer.o: $(inc)/timer.cpp $(inc)/timer.h 
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cpp $(inc)/kernels.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) $(inc)/data/dirty_4096.img $(inc)/data/psf_4096.img 1000
