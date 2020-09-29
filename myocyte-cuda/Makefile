# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
myocyte.out:		main.o
	nvcc	main.o \
				-I/usr/local/cuda/include \
				-L/usr/local/cuda/lib \
				-lm \
        -o myocyte.out

# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work.cu \
				solver.cu \
				embedded_fehlberg_7_8.cu \
				master.cu \
				kernel.cu \
				kernel_ecc.cu \
				kernel_cam.cu \
				kernel_fin.cu \
				file.c \
				timer.c
	nvcc	main.cu \
				-c -O3 -g -DDEBUG

run: myocyte.out
	./myocyte.out 100 

# delete all object files
clean:
	rm -rf *.o *.out output.txt

