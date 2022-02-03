include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp Option.cpp OptionParser.cpp S3D.cpp 

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp OptionParser.h Utility.h 

Option.o: Option.cpp Option.h 

OptionParser.o: OptionParser.cpp OptionParser.h Utility.h

S3D.o: S3D.cpp S3D.h OptionParser.h *.sycl

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -q -n $(NUM_PASSES) -s 1
