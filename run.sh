# For SYCL
export SYCL_BE=PI_OPENCL

for dir in `find . -mindepth 1 -maxdepth 1 -type d | grep -Ev '.\.git|include|cuda|omp'`
do
	cd ${dir}
	make clean
	cliloader -q -h -d make run &> report.txt
	echo "${dir} results:"
	grep "Total Enqueues" report.txt
	grep "Total Time" report.txt 
	cd ..
done

# For OMP target offloading
export LIBOMPTARGET_PLUGIN=OPENCL

for dir in `find . -mindepth 1 -maxdepth 1 -type d | grep -Ev '.\.git|include|sycl|dpct'`
do
	cd ${dir}
	make clean
	cliloader -q -h -d make run &> report.txt
	echo "${dir} results:"
	grep "Total Enqueues" report.txt
	grep "Total Time" report.txt 
	cd ..
done
