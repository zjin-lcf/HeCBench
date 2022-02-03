include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp galois.cpp jerasure.cpp GCRSMatrix.cpp utils.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

jerasure.o: jerasure.h galois.h
galois.o: galois.h
GCRSMatrix.o: GCRSMatrix.h jerasure.h galois.h
main.o: utils.h GCRSMatrix.h kernels.cpp
utils.o: utils.h

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) 10 10
