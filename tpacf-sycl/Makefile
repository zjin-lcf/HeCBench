include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o args.o compute.o

#===============================================================================
# Runtime Variables
#===============================================================================
POINTS    = 97178
DATAPARS  = -d ../tpacf-cuda/data/small/Datapnts.1 -p $(POINTS)
RANDPARS  = -r ../tpacf-cuda/data/small/Randompnts -n 100 -q $(POINTS)
ANALPARS  = -b 5 -l 1 -u 10000 -a -j 10
OUTFILE   = ./small_results_test_g200.dat

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

args.o: ../tpacf-cuda/args.c ../tpacf-cuda/args.h
	$(CC) $(CFLAGS) -c $< -o $@

compute.o: compute.cpp model_io.cpp ACF_kernel.cpp histogram_kernel.cpp ../tpacf-cuda/args.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp ../tpacf-cuda/args.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) $(OUTFILE)

run: $(program)
	./$(program) $(DATAPARS) $(RANDPARS) $(ANALPARS) -o $(OUTFILE)
