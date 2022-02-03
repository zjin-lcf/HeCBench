include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================


#===============================================================================
# Targets to Build
#===============================================================================

./srad:	./main.o \
	./kernel/kernel_wrapper.o \
	./util/graphics/graphics.o \
	./util/graphics/resize.o \
	./util/timer/timer.o
	$(CC) $(CFLAGS)	./main.o \
		./kernel/kernel_wrapper.o \
		./util/graphics/graphics.o \
		./util/graphics/resize.o \
		./util/timer/timer.o \
		-lm \
		$(LDFLAGS) -o srad

./main.o:./main.c ./main.h 
	$(CC)	$(CFLAGS) $(KERNEL_DIM) ./main.c \
		-c \
		-o ./main.o \
		-O3

./kernel/kernel_wrapper.o:./kernel/kernel_wrapper.h \
	./kernel/kernel_wrapper.c \
	./kernel/kernel_compress.sycl \
	./kernel/kernel_extract.sycl \
	./kernel/kernel_prepare.sycl \
	./kernel/kernel_reduce.sycl \
	./kernel/kernel_srad.sycl \
	./kernel/kernel_srad2.sycl 
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o 

./util/graphics/graphics.o:./util/graphics/graphics.h ./util/graphics/graphics.c
	$(CC)	./util/graphics/graphics.c \
		-c \
		-o ./util/graphics/graphics.o \
		-O3

./util/graphics/resize.o:./util/graphics/resize.h ./util/graphics/resize.c
	$(CC)	./util/graphics/resize.c \
		-c \
		-o ./util/graphics/resize.o \
		-O3

./util/timer/timer.o:./util/timer/timer.h ./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

clean:
	rm	./*.o srad *.pgm\
		./kernel/*.o \
		./util/graphics/*.o \
		./util/timer/*.o

run: srad
	./srad 1000 0.5 502 458

