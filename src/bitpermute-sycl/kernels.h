// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __NTT_KERNELS_CU__
#define __NTT_KERNELS_CU__

#define WARP_SZ 32
#define MAX_LG_DOMAIN_SIZE 28

#if MAX_LG_DOMAIN_SIZE <= 32
typedef unsigned int index_t;
#else
typedef size_t index_t;
#endif

#define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 1) / 2)
#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)

//typedef int fr_t;
typedef long fr_t;


/// Reverse the bit order of an unsigned integer
/// \param [in] a Input unsigned integer value
/// \returns Value of a with the bit order reversed
template <typename T>
inline T reverse_bits(T a) {
  static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value,
                "unsigned integer required");
  if (!a)
    return 0;
  T mask = 0;
  size_t count = 4 * sizeof(T);
  mask = ~mask >> count;
  while (count) {
    a = ((a & mask) << count) | ((a & ~mask) >> count);
    count = count >> 1;
    mask = mask ^ (mask << count);
  }
  return a;
}


template <typename T> inline T bit_rev(T i, unsigned int nbits)
{
    if (sizeof(i) == 4 || nbits <= 32)
        return reverse_bits<unsigned int>(i) >>
               (8 * sizeof(unsigned int) - nbits);
    else
        return reverse_bits<unsigned long long>(i) >>
               (8 * sizeof(unsigned long long) - nbits);
}

// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
//__launch_bounds__(1024) 

void bit_rev_permutation(fr_t* d_out, const fr_t *d_in, uint32_t lg_domain_size,
                         const sycl::nd_item<1> &item)
{
    if (item.get_group_range(0) == 1 &&
        item.get_local_range(0) == (1 << lg_domain_size)) {
        uint32_t idx = item.get_local_id(0);
        uint32_t rev = bit_rev(idx, lg_domain_size);

        fr_t t = d_in[idx];
        if (d_out == d_in)
            item.barrier(sycl::access::fence_space::local_space);
        d_out[rev] = t;
    } else {
        index_t idx = 
            item.get_local_id(0) +
            item.get_local_range(0) * (index_t)item.get_group(0);
        index_t rev = bit_rev(idx, lg_domain_size);
        bool copy = d_out != d_in && idx == rev;

        if (idx < rev || copy) {
            fr_t t0 = d_in[idx];
            if (!copy) {
                fr_t t1 = d_in[rev];
                d_out[idx] = t1;
            }
            d_out[rev] = t0;
        }
    }
}

constexpr int BLOCK_SIZE = 192;

template<typename T>
static constexpr uint32_t lg2(T n)
{   uint32_t ret=0; while (n>>=1) ret++; return ret;   }

//__launch_bounds__(BLOCK_SIZE, 2)
void bit_rev_permutation_z(fr_t* out, const fr_t* in, uint32_t lg_domain_size,
                           const sycl::nd_item<1> &item, uint8_t *sm)
{
    const uint32_t Z_COUNT = 256 / sizeof(fr_t);
    const uint32_t LG_Z_COUNT = lg2(Z_COUNT);

    auto exchange = (fr_t *)sm;
    fr_t (*xchg)[Z_COUNT][Z_COUNT] = reinterpret_cast<decltype(xchg)>(exchange);

    uint32_t gid = item.get_local_id(0) / Z_COUNT;
    uint32_t idx = item.get_local_id(0) % Z_COUNT;
    uint32_t rev = bit_rev(idx, LG_Z_COUNT);

    index_t step = (index_t)1 << (lg_domain_size - LG_Z_COUNT);
    index_t tid = item.get_local_id(0) +
                  item.get_local_range(0) * (index_t)item.get_group(0);

    auto g = item.get_group();
    auto sg = item.get_sub_group();

#pragma unroll 1
    do {
        index_t group_idx = tid >> LG_Z_COUNT;
        index_t group_rev = bit_rev(group_idx, lg_domain_size - 2*LG_Z_COUNT);

        if (group_idx > group_rev)
            continue;

        index_t base_idx = group_idx * Z_COUNT + idx;
        index_t base_rev = group_rev * Z_COUNT + idx;

        fr_t regs[Z_COUNT];

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            xchg[gid][i][rev] = (regs[i] = in[i * step + base_idx]);
            if (group_idx != group_rev)
                regs[i] = in[i * step + base_rev];
        }

        (Z_COUNT > WARP_SZ) ?  sycl::group_barrier(g) : sycl::group_barrier(sg);

#pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_rev] = xchg[gid][rev][i];

        if (group_idx == group_rev)
            continue;

        (Z_COUNT > WARP_SZ) ?  sycl::group_barrier(g) : sycl::group_barrier(sg);

#pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            xchg[gid][i][rev] = regs[i];

        (Z_COUNT > WARP_SZ) ?  sycl::group_barrier(g) : sycl::group_barrier(sg);

#pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_idx] = xchg[gid][rev][i];

    } while (Z_COUNT <= WARP_SZ && (tid += item.get_local_range(0) *
                                           item.get_group_range(0)) < step);
    // without "Z_COUNT <= WARP_SZ" compiler spills 128 bytes to stack :-(
}

#endif /* __NTT_KERNELS_CU__ */
