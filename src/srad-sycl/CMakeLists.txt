# srad-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL sycl
    SOURCES main.cpp
            util/graphics/graphics.c
            util/graphics/resize.c
	    util/timer/timer.c
    CATEGORIES algorithms
)
