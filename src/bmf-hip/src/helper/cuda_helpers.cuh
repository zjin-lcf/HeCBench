#ifndef HIP_HELPERS_CUH
#define HIP_HELPERS_CUH

#include <iostream>
#include <cstdint>

#include "../config.h"

#include <chrono>

#define TIMERSTART(label)                                                  \
    std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
    a##label = std::chrono::system_clock::now();

#define TIMERSTOP(label)                                                   \
    b##label = std::chrono::system_clock::now();                           \
    std::chrono::duration<double> delta##label = b##label-a##label;        \
    std::cout << "# elapsed time ("<< #label <<"): "                       \
              << delta##label.count()  << "s" << std::endl;

#ifdef __HIPCC__
    #define CUERR {                                                            \
        hipError_t err;                                                       \
        if ((err = hipGetLastError()) != hipSuccess) {                       \
            std::cout << "HIP error: " << hipGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }
#endif

#ifdef __HIPCC__
    #define HOST_DEVICE_QUALIFIER __host__ __device__
#else
    #define HOST_DEVICE_QUALIFIER 
#endif

// safe division
#ifndef SDIV
    #define SDIV(x,y)(((x)+(y)-1)/(y))
#endif

// floor to next multiple of y
#ifndef FLOOR
    #define FLOOR(x,y)(((x)/(y)*(y))
#endif

#define FULLMASK 0xffffffff


#endif
