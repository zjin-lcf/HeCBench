#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <iostream>
#include <cstdint>
#include <chrono>
#include <sycl/sycl.hpp>

#include "../config.h"

#define TIMERSTART(label)                                                  \
    std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
    a##label = std::chrono::system_clock::now();

#define TIMERSTOP(label)                                                   \
    b##label = std::chrono::system_clock::now();                           \
    std::chrono::duration<double> delta##label = b##label-a##label;        \
    std::cout << "# elapsed time ("<< #label <<"): "                       \
              << delta##label.count()  << "s" << std::endl;

#define HOST_DEVICE_QUALIFIER 

// safe division
#ifndef SDIV
    #define SDIV(x,y)(((x)+(y)-1)/(y))
#endif

// floor to next multiple of y
#ifndef FLOOR
    #define FLOOR(x,y)(((x)/(y)*(y))
#endif

#define FULLMASK 0xFFFFFFFF

#endif
