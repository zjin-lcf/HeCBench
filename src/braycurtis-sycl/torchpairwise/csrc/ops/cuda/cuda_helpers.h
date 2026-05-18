#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t)                        \
    for (index_t i = item.get_global_id(2);                         \
         i < (n);                                                   \
         i += (item.get_local_range(2) * item.get_group_range(2)))

#define CUDA_1D_KERNEL_LOOP(i, n) \
CUDA_1D_KERNEL_LOOP_T(i, n, std::remove_cv_t<std::remove_reference_t<decltype(n)>>)

inline unsigned int GET_BLOCKS(const unsigned int THREADS,
                               const unsigned int N)
{
    unsigned int kMaxGridNum = (N + THREADS - 1) / THREADS;
    //return std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
}

// Temporarily counter latest MSVC update that causes incompatibility with CUDA
#if (_MSC_VER >= 1928)
#define floor floorf
#define ceil ceilf
#endif
