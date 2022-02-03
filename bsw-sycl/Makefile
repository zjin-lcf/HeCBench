include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp driver.cpp utils.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp kernel.cpp driver.hpp kernel.hpp utils.hpp 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) \
                ../bsw-cuda/test-data/protein-reference.fasta \
                ../bsw-cuda/test-data/protein-query.fasta test_regression_protein_out 
	diff test_regression_protein_out ../bsw-cuda/test-data/result_aa
