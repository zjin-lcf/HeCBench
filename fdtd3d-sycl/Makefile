include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp FDTD3dReference.cpp FDTD3dGPU.cpp shrUtils.cpp cmd_arg_reader.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp shrUtils.h cmd_arg_reader.h exception.h FDTD3dGPU.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) --dimx=192 --dimy=184 --timesteps=90

