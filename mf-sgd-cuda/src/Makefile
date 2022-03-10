
CU_FLAG=-gencode arch=compute_35,code=compute_35 \
	-gencode arch=compute_37,code=compute_37 \
        -gencode arch=compute_50,code=compute_50 \
        -gencode arch=compute_60,code=compute_60 \
	-O3

mf_sgd:main.o sgd.o
	nvcc $(CU_FLAG) main.o sgd.o -o mf_sgd -Xcompiler -fopenmp -DUSEOMP

main.o:main.cpp sgd.h 
	nvcc $(CU_FLAG) -c main.cpp -std=c++11 -DUSEOMP -Xcompiler -fopenmp

sgd.o:sgd.cu sgd.h
	nvcc $(CU_FLAG) -c sgd.cu -o sgd.o -std=c++11 -Xcompiler -fopenmp -DUSEOMP

clean:
	rm -f mf_sgd *.o
