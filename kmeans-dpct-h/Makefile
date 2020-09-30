CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no

# Standard flags
CFLAGS := -std=c++14 -Wall


# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

all: kmeans

kmeans: cluster.o getopt.o read_input.o rmse.o kmeans.cpp
	$(CC) $(KERNEL_DIM) $(CFLAGS) $^ -o kmeans $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) $< -c

cluster.o: cluster.cpp kmeans.h

getopt.o: getopt.cpp getopt.h

read_input.o: read_input.cpp kmeans.h

rmse.o: rmse.cpp kmeans.h

clean:
	rm -f *.o *~ kmeans *.linkinfo

run: kmeans
	./kmeans -r -n 5 -m 15 -l 10 -o -i ../data/kmeans/kdd_cup


