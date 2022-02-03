include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp curvilinear4sg.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp utils.cpp utils.h 
	$(CC) $(CFLAGS) -c $< -o $@

curvilinear4sg.o : curvilinear4sg.cpp \
       	           kernel1.cpp \
	           kernel2.cpp \
       	           kernel3.cpp \
	           kernel4.cpp \
	           kernel5.cpp \
	           utils.h \
	           curvilinear4sg.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) ../sw4ck-cuda/sw4ck.in 100
