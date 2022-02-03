include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

c_source = main.c fbase.c fparam.c util.c
sycl_source = frna.cpp 

obj = $(c_source:.c=.o) $(sycl_source:.cpp=.o) 

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

fbase.o : fbase.c fbase.h util.h cu.h
main.o : main.c frna.h util.h fbase.h
fparam.o : fparam.c fparam.h util.h fbase.h int.h
fprna.o : fprna.cpp frna.h util.h cu.h
util.o : util.c util.h

clean:
	rm -rf $(program) $(obj) *.out

test: $(program)
	./$(program) ../frna-cuda/RD0260.seq rd0260.out

run: $(program)
	./$(program) ../prna-cuda/HIV1-NL43.seq  hiv1-nl43.out
