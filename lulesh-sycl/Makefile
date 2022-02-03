include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = lulesh

source = lulesh.cc \
        lulesh-viz.cc \
        lulesh-util.cc \
        lulesh-init.cc

obj = $(source:.cc=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cc lulesh.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) -i 1 -s 32 -r 11 -b 1 -c 1

run: $(program)
	./$(program) -i $(NUM_PASSES) -s 128 -r 11 -b 1 -c 1


