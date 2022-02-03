include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp ./util/num/num.cpp ./util/timer/timer.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o:	main.cpp main.h kernel.sycl

./util/num/num.o: ./util/num/num.cpp ./util/num/num.h 

./util/timer/timer.o: ./util/timer/timer.cpp ./util/timer/timer.h 

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -boxes1d 30
