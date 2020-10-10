CC        = dpcpp
OPTIMIZE  = yes
DEBUG     = no



# Standard flags
CFLAGS := -std=c++11 -Wall

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



ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

hybridsort: hybridsort.o bucketsort.o mergesort.o
	$(CC) $(CFLAGS) -o hybridsort $(VERIFY) $(OUTPUT) $(TIMER) \
		hybridsort.o bucketsort.o mergesort.o $(LDFLAGS)

hybridsort.o : hybridsort.cpp mergesort.h bucketsort.h
	$(CC) $(CFLAGS) -o hybridsort.o -c hybridsort.cpp

bucketsort.o: bucketsort.cpp \
	kernel_bucketprefix.h \
	kernel_histogram.h \
	kernel_bucketcount.h \
       	kernel_bucketsort.h  
	$(CC) $(CFLAGS) -o bucketsort.o -c bucketsort.cpp 

mergesort.o: mergesort.cpp mergesort.h kernel_mergeSortPass.h
	$(CC) $(CFLAGS) -o mergesort.o -c mergesort.cpp

run: hybridsort
	./hybridsort r

clean:
	rm -f  *.o hybridsort
