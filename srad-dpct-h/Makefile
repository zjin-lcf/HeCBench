CC := dpcpp

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
srad:		main.o
	$(CC)	main.o \
				-lm -o srad

# compile main function file into object (binary)
main.o: 	main.dp.cpp \
				main.h \
				graphics.c \
				extract_kernel.dp.cpp \
				prepare_kernel.dp.cpp \
				reduce_kernel.dp.cpp \
				srad_kernel.dp.cpp \
				srad2_kernel.dp.cpp \
				compress_kernel.dp.cpp
	$(CC)	main.dp.cpp \
				-c -Wall -O3 -o main.o

# delete all object files
clean:
	rm -f *.o srad *pgm

run: srad
	./srad 1000 0.5 502 458
