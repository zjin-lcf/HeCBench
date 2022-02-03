include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = euler3d.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp kernel_compute_flux.sycl  \
           kernel_compute_step_factor.sycl  \
	   kernel_initialize_variables.sycl \
	   kernel_memset.sycl \
	   kernel_time_step.sycl
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) result* density momentum density_energy

run: $(program)
	./$(program) ../data/cfd/fvcorr.domn.097K 
	#./$(program) ../data/cfd/fvcorr.domn.193K   
	#./$(program) ../data/cfd/missile.domn.0.2M  
