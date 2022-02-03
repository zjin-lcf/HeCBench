include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = KeccakF.cpp KeccakTreeCPU.cpp KeccakTreeGPU.cpp Test.cpp main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

KeccakTreeCPU.o: KeccakTreeCPU.cpp KeccakTreeCPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

KeccakTreeGPU.o: KeccakTreeGPU.cpp KeccakTreeGPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

Test.o: Test.cpp KeccakTreeCPU.h KeccakTreeGPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp KeccakTreeCPU.h KeccakTreeGPU.h Test.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

KeccakF.o: KeccakF.cpp KeccakTree.h KeccakF.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program)
