include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================


#===============================================================================
# Targets to Build
#===============================================================================

./myocyte.out:./main.o \
              ./kernel/kernel_wrapper.o \
              ./util/file/file.o \
              ./util/num/num.o \
              ./util/timer/timer.o 
	$(CC) $(CFLAGS)	./main.o \
		./kernel/kernel_wrapper.o \
		./util/file/file.o \
		./util/num/num.o \
		./util/timer/timer.o \
		$(LDFLAGS) \
		-o myocyte.out

./main.o:./common.h ./main.h ./main.c
	$(CC) $(CFLAGS)	./main.c \
		-c \
		-o ./main.o 

./kernel/kernel_wrapper.o:./common.h \
	./kernel/kernel_wrapper.h \
	./kernel/kernel_wrapper.c \
	./kernel/solver.c \
	./kernel/embedded_fehlberg_7_8.c \
	./kernel/master.c \
	./kernel/kernel_cam.sycl \
	./kernel/kernel_ecc.sycl
	$(CC) $(CFLAGS)	./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o 

./util/file/file.o:./common.h \
	./util/file/file.h \
	./util/file/file.c
	$(CC)	./util/file/file.c \
		-c \
		-o ./util/file/file.o \
		-O3

./util/num/num.o:./common.h \
	./util/num/num.h \
	./util/num/num.c
	$(CC)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

./util/timer/timer.o:./common.h \
	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

clean:
	rm	*.o *.out \
		./kernel/*.o \
		./util/file/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		./output.txt

run: myocyte.out
	./myocyte.out -time 100
