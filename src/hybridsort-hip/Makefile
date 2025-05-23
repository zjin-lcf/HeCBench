CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g #-DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

main: hybridsort.o bucketsort.o mergesort.o
	$(CC) $(CFLAGS) -o main $(VERIFY) $(OUTPUT) $(TIMER) \
		hybridsort.o bucketsort.o mergesort.o $(LDFLAGS)

hybridsort.o: hybridsort.cu bucketsort.h mergesort.h
	$(CC) $(CFLAGS) -o hybridsort.o -c hybridsort.cu

bucketsort.o: bucketsort.cu \
	kernel_bucketprefix.h \
	kernel_histogram.h \
	kernel_bucketcount.h \
       	kernel_bucketsort.h  
	$(CC) $(CFLAGS) -o bucketsort.o -c bucketsort.cu

mergesort.o: mergesort.cu mergesort.h kernel_mergeSortPass.h
	$(CC) $(CFLAGS) -o mergesort.o -c mergesort.cu

run: hybridsort
	$(LAUNCHER) ./main r

clean:
	rm -f *.o main
