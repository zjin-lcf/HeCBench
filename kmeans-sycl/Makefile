include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = cluster.c getopt.c read_input.c rmse.c kmeans.c

obj = $(source:.c=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

kmeans.o: kmeans.c kmeans.h

cluster.o: cluster.c kmeans.h

getopt.o: getopt.c getopt.h

read_input.o: read_input.c kmeans.h

rmse.o: rmse.c kmeans.h

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -r -n 5 -m 15 -l 10 -o -i ../data/kmeans/kdd_cup
