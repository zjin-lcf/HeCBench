include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp Option.cpp OptionParser.cpp QTC.cpp comm.cpp libdata.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp OptionParser.h Utility.h 
	$(CC) $(CFLAGS) -c $< -o $@

comm.o: comm.cpp comm.h
	$(CC) $(CFLAGS) -c $< -o $@

libdata.o: libdata.cpp libdata.h
	$(CC) $(CFLAGS) -c $< -o $@

Option.o: Option.cpp Option.h 
	$(CC) $(CFLAGS) -c $< -o $@

OptionParser.o: OptionParser.cpp OptionParser.h Utility.h
	$(CC) $(CFLAGS) -c $< -o $@

QTC.o: QTC.cpp QTC.h OptionParser.h libdata.h \
       tuningParameters.h comm.h kernels_compact_storage.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) --Verbose
	#./$(program) --SaveOutput

