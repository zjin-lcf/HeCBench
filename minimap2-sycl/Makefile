include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = device/device_kernel_wrapper.cpp \
         src/host_data_io.cpp   \
         src/host_kernel.cpp   \
         src/main.cpp   \
         src/memory_scheduler.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp device/device_kernel.cpp \
           include/datatypes.h include/device_kernel_wrapper.h \
	   include/host_data_io.h include/host_kernel.h \
	   include/kernel_common.h include/memory_scheduler.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) in-1k.txt kernel-out.txt
