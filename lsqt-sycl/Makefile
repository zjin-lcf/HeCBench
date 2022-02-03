include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp lsqt.cpp sigma.cpp vector.cpp hamiltonian.cpp model.cpp \
         model_general.cpp model_lattice.cpp anderson.cpp charge.cpp

header = sigma.h vector.h lsqt.h hamiltonian.h model.h anderson.h charge.h

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.out

run: $(program)
	./$(program) ../lsqt-cuda/examples/input.txt
