CC := nvcc

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
main.o: 	main.cu \
				main.h \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	$(CC)	main.cu \
				-c -Xcompiler -Wall -O3 -arch sm_35

# delete all object files
clean:
	rm -f *.o srad *pgm

run: srad
	./srad 1000 0.5 502 458
