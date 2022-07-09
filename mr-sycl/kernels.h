#include "common.h"

void mr32_sf(
  nd_item<1> &item,
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  int j = item.get_global_id(0);
  if (j < iter) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      // atomicAdd(val, straightforward_mr32(bases, cnt, n));
      auto ao = ext::oneapi::atomic_ref<int, 
            ext::oneapi::memory_order::relaxed,
            ext::oneapi::memory_scope::device,
            access::address_space::global_space> (val[0]);
      ao.fetch_add(straightforward_mr32(bases, cnt, n));
    }
  }
}

void mr32_eff(
  nd_item<1> &item,
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  int j = item.get_global_id(0);
  if (j < iter) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      // atomicAdd(val, efficient_mr32(bases, cnt, n));
      auto ao = ext::oneapi::atomic_ref<int, 
            ext::oneapi::memory_order::relaxed,
            ext::oneapi::memory_scope::device,
            access::address_space::global_space> (val[0]);
      ao.fetch_add(efficient_mr32(bases, cnt, n));
    }
  }
}
