inline int atomicAdd(int *var, int val)
{
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*var);
  return atm.fetch_add(val);
}

void mr32_sf(
  sycl::nd_item<1> &item,
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  int j = item.get_global_id(0);
  if (j < iter) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      atomicAdd(val, straightforward_mr32(bases, cnt, n));
    }
  }
}

void mr32_eff(
  sycl::nd_item<1> &item,
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  int j = item.get_global_id(0);
  if (j < iter) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      atomicAdd(val, efficient_mr32(bases, cnt, n));
    }
  }
}
