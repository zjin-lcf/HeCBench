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

%.o: %.cpp black_kernel.sycl calculate_F.sycl calculate_u.sycl set_BCs.sycl \
 	set_vert_pres_BCs.sycl calc_residual.sycl calculate_G.sycl calculate_v.sycl  \
	red_kernel.sycl set_horz_pres_BCs.sycl sum_pressure.sycl
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.dat

run: $(program)
	./$(program)

