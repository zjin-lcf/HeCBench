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

heartwall:./main.o \
          ./kernel/kernel.o \
          ./util/avi/avilib.o \
          ./util/avi/avimod.o \
          ./util/file/file.o \
          ./util/timer/timer.o 
	$(CC)	$(CFLAGS) ./main.o \
		./kernel/kernel.o \
		./util/avi/avilib.o \
		./util/avi/avimod.o \
		./util/file/file.o \
		./util/timer/timer.o \
		$(LDFLAGS) -o heartwall

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:./main.h ./main.c
	$(CC)	$(OUTPUT) $(KERNEL_DIM) ./main.c \
		-I./util/timer/ \
		-I./util/file/ \
		-c \
		-o ./main.o 

./kernel/kernel.o:./kernel/kernel.sycl ./kernel/kernel.cpp
	$(CC) $(CFLAGS) $(TEST_ON) $(KERNEL_DIM) ./kernel/kernel.cpp \
		-c \
		-o ./kernel/kernel.o 

./util/avi/avilib.o:./util/avi/avilib.h ./util/avi/avilib.c
	$(CC)	$(CFLAGS) ./util/avi/avilib.c \
		-I./util/avi \
		-c \
		-o ./util/avi/avilib.o 

./util/avi/avimod.o:./util/avi/avilib.h ./util/avi/avimod.c
	$(CC)	$(CFLAGS) ./util/avi/avimod.c \
		-I./util/avi \
		-c \
		-o ./util/avi/avimod.o 

./util/file/file.o:./util/file/file.h ./util/file/file.c
	$(CC)	$(CFLAGS) ./util/file/file.c \
		-I./util/file \
		-I./util/timer \
		-c \
		-o ./util/file/file.o 

./util/timer/timer.o:./util/timer/timer.h ./util/timer/timer.c
	$(CC)	$(CFLAGS) ./util/timer/timer.c \
		-I./util/timer \
		-c \
		-o ./util/timer/timer.o 

clean:
	rm	-rf *.o heartwall \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o 

run: heartwall
	./heartwall 104

