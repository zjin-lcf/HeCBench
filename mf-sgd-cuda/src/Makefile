


CU_FLAG=-gencode arch=compute_35,code=compute_35 \
	-gencode arch=compute_37,code=compute_37 \
        -gencode arch=compute_50,code=compute_50 \
        -gencode arch=compute_60,code=compute_60 \
	-O3

cumf_sgd:main.o sgd.o sgd_kernel.o
	nvcc $(CU_FLAG) main.o sgd.o sgd_kernel.o -o cumf_sgd -Xcompiler -fopenmp -DUSEOMP

main.o:main.cpp sgd.h 
	nvcc $(CU_FLAG) -c main.cpp -std=c++11 -DUSEOMP -Xcompiler -fopenmp

sgd.o:sgd.cu sgd.h
	nvcc -c sgd.cu -o sgd.o -std=c++11 -Xcompiler -fopenmp -DUSEOMP $(CU_FLAG)

sgd_kernel.o:sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -c sgd_kernel.cu -o sgd_kernel.o -std=c++11 -Xcompiler -fopenmp -DUSEOMP $(CU_FLAG) -Xptxas -v

clean:
	rm -f cumf_sgd *.o
