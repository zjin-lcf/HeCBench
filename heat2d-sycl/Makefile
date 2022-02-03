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

%.o: %.cpp io.c lapl_ss.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

data.txt: mkinit.py
	python2 mkinit.py 4096 8192 data.txt

run: $(program) data.txt
	./$(program) 4096 8192 $(NUM_PASSES) data.txt

