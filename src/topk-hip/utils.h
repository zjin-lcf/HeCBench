#pragma once

#include <hipcub/hipcub.hpp>
//#include <hipcub/util_type.hpp>

#define GPU_CHECK(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while (0)



