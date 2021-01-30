VENDOR      = intel
OPTIMIZE    = yes
DEBUG       = no



# Standard flags
CFLAGS := -std=c++11 -Wall

# Linker flags
LDFLAGS = -lm

# Codeplay Compiler
ifneq (,$(findstring codeplay,$(VENDOR)))
  CC = compute++
  CFLAGS += -no-serial-memop -sycl -sycl-driver
  LDFLAGS += -lComputeCpp
endif

# Intel DPC++ Compiler
ifneq (,$(findstring intel,$(VENDOR)))
  CC = dpcpp
  CFLAGS += 
  LDFLAGS += 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


b+tree.out:	./main.o \
		./kernel/kernel_wrapper.o \
		./kernel/kernel2_wrapper.o \
		./util/timer/timer.o \
		./util/num/num.o 
	$(CC)	./main.o \
			./kernel/kernel_wrapper.o \
			./kernel/kernel2_wrapper.o \
			./util/timer/timer.o \
			./util/num/num.o \
			$(LDFLAGS) \
	                -o b+tree.out


main.o:	./common.h \
	./main.cpp
	$(CC) $(CFLAGS) $(KERNEL_DIM) ./main.cpp \
		-c \
		-o ./main.o


./kernel/kernel_wrapper.o:	./common.h \
	./kernel/kernel_wrapper.h \
	./kernel/kernel_wrapper.cpp \
	./kernel/kernel.cpp
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_wrapper.cpp \
		-c \
		-o ./kernel/kernel_wrapper.o

./kernel/kernel2_wrapper.o:	./common.h \
	./kernel/kernel2_wrapper.h \
	./kernel/kernel2_wrapper.cpp \
	./kernel/kernel2.cpp
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel2_wrapper.cpp \
		-c \
		-o ./kernel/kernel2_wrapper.o 


./util/timer/timer.o:	./common.h \
	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

./util/num/num.o:	./common.h \
	./util/num/num.h \
	./util/num/num.c
	$(CC)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

run: b+tree.out
	./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt

clean:
	rm -f	*.o *.out \
		./kernel/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		output.txt
