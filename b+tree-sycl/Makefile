include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================


#===============================================================================
# Targets to Build
#===============================================================================

b+tree.out:./main.o \
           ./kernel/kernel_wrapper.o \
           ./kernel/kernel2_wrapper.o \
           ./util/timer/timer.o \
           ./util/num/num.o 
	$(CC) $(CFLAGS)	./main.o \
			./kernel/kernel_wrapper.o \
			./kernel/kernel2_wrapper.o \
			./util/timer/timer.o \
			./util/num/num.o \
			$(LDFLAGS) \
	                -o b+tree.out


main.o:	./b+tree.h ./main.c
	$(CC) $(CFLAGS) $(KERNEL_DIM) ./main.c -c -o main.o


./kernel/kernel_wrapper.o: ./b+tree.h \
                    ./kernel/kernel_wrapper.h \
                    ./kernel/kernel_wrapper.c \
                    ./kernel/findK.sycl
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o

./kernel/kernel2_wrapper.o: ./b+tree.h \
                   ./kernel/kernel2_wrapper.h \
                   ./kernel/kernel2_wrapper.c \
                   ./kernel/findRangeK.sycl
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel2_wrapper.c \
		-c \
		-o ./kernel/kernel2_wrapper.o 


./util/timer/timer.o: ./b+tree.h \
                  ./util/timer/timer.h \
                  ./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

./util/num/num.o: ./b+tree.h \
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
