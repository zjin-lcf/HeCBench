# srad-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL omp
    SOURCES main.c
            util/graphics/graphics.c
            util/graphics/resize.c
	    util/timer/timer.c
    CATEGORIES algorithms
    LINK_LIBRARIES m
)
