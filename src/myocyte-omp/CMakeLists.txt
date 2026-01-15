# myocyte-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME myocyte
    MODEL omp
    SOURCES main.c
	    kernel/kernel_wrapper.c
	    util/file/file.c
	    util/num/num.c
    CATEGORIES algorithms
)
set_source_files_properties(main.c kernel/kernel_wrapper.c util/file/file.c  util/num/num.c PROPERTIES LANGUAGE CXX)
