include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = AMGMk

source = csr_matrix.cpp  csr_matvec.cpp  hypre_error.cpp  hypre_memory.cpp  laplace.cpp  main.cpp  vector.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)

