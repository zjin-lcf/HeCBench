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

$(obj): main.cpp \
       kernel_CopySrcToComponent.sycl \
       kernel_CopySrcToComponents.sycl \
       kernel_fdwt53.sycl dwt.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)
	rm -rf *.bmp.dwt.*

run: $(program)
	#./$(program) 192.bmp -d 192x192 -f -5 -l 3
	./$(program) rgb.bmp -d 1024x1024 -f -5 -l 3
