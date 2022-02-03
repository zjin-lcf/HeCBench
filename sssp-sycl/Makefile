include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp kernel.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp kernel.h support/common.h support/timer.h support/verify.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# 120 work-groups, 1 CPU thread, 10 warm-up runs, and 1 timed run
run: $(program)
	./$(program) -g 120 -t 1 -w 10 -r 1
