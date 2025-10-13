#define ATOMIC_REF(v) \
  sycl::atomic_ref<T, sycl::memory_order::relaxed, \
                   sycl::memory_scope::device,\
                   sycl::access::address_space::generic_space>(v)

template <class T>
void testKernel(sycl::nd_item<1> &item, T *g_odata, size_t len)
{
  const size_t i = item.get_global_id(0);
  if (i >= len) return;

  auto ao0 = ATOMIC_REF(g_odata[0]);
  ao0.fetch_add((T)10);

  auto ao1 = ATOMIC_REF(g_odata[1]);
  ao1.fetch_sub((T)10);

  auto ao2 = ATOMIC_REF(g_odata[2]);
  ao2.fetch_max((T)i);

  auto ao3 = ATOMIC_REF(g_odata[3]);
  ao3.fetch_min((T)i);

  auto ao4 = ATOMIC_REF(g_odata[4]);
  ao4.fetch_and((T)(2*i+7));

  auto ao5 = ATOMIC_REF(g_odata[5]);
  ao5.fetch_or((T)(1<<i));

  auto ao6 = ATOMIC_REF(g_odata[6]);
  ao6.fetch_xor((T)(i));

  // atomicInc and atomicDec are not fully supported across
  // vendors' GPUs. The implementations are from Syclomatic.
  auto ao7 = ATOMIC_REF(g_odata[7]);
  while (true) {
    T old = ao7.load();
    if (old >= 17) {
      if (ao7.compare_exchange_strong(old, 0))
        break;
    } else if (ao7.compare_exchange_strong(old, old + 1))
      break;
  }

  auto ao8 = ATOMIC_REF(g_odata[8]);
  while (true) {
    T old = ao8.load();
    if (old <= 0) {
      if (ao8.compare_exchange_strong(old, 137))
        break;
    } else if (ao8.compare_exchange_strong(old, old - 1))
      break;
  }
}

