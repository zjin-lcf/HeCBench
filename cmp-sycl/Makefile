include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

inc = ../cmp-cuda

obj = main.o reference.o su_cdp.o su_gather.o su_trace.o log.o parser.o

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

log.o : $(inc)/log.cpp $(inc)/log.hpp
parser.o: $(inc)/parser.cpp $(inc)/parser.hpp  
su_cdp.o : $(inc)/su_cdp.cpp $(inc)/su_cdp.hpp $(inc)/log.hpp
su_gather.o : $(inc)/su_gather.cpp $(inc)/su_gather.hpp $(inc)/log.hpp $(inc)/su_trace.hpp
su_trace.o : $(inc)/su_trace.cpp $(inc)/su_trace.hpp $(inc)/log.hpp
main.o : main.cpp $(inc)/log.hpp $(inc)/utils.hpp $(inc)/parser.hpp $(inc)/su_gather.hpp
reference.o : $(inc)/reference.cpp $(inc)/utils.hpp

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -c0 1.98e-7 -c1 1.77e-6 -nc 5 -aph 600 -tau 0.002 -v 2 \
                     -i $(inc)/data/simple-synthetic.su
