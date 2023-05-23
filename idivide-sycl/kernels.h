template<typename divisor_type>
void throughput_test(
    sycl::nd_item<1> &item,
    divisor_type d1,
    divisor_type d2,
    divisor_type d3,
    int dummy,
    int * buf)
{
  int x = item.get_global_id(0);
  int x1 = x / d1;
  int x2 = x / d2;
  int x3 = x / d3;
  int aggregate = x1 + x2 + x3;  
  if (aggregate & dummy == 1) buf[0] = aggregate;
}

template<typename divisor_type>
void latency_test(
    sycl::nd_item<1> &item,
    divisor_type d1,
    divisor_type d2,
    divisor_type d3,
    divisor_type d4,
    divisor_type d5,
    divisor_type d6,
    divisor_type d7,
    divisor_type d8,
    divisor_type d9,
    divisor_type d10,
    int dummy,
    int * buf)
{
  int x = item.get_global_id(0);
  x /= d1;
  x /= d2;
  x /= d3;
  x /= d4;
  x /= d5;
  x /= d6;
  x /= d7;
  x /= d8;
  x /= d9;
  x /= d10;
  if (x & dummy == 1) buf[0] = x;
}

inline int atomicFetchAdd(int &val, const int delta)
{
  sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(val);
  return ref.fetch_add(delta);
}

void check(sycl::nd_item<1> &item, int_fastdiv divisor, int * results)
{
  int divident = item.get_global_id(0);

  int quotient = divident / (int)divisor;
  int fast_quotient = divident / divisor;

  if (quotient != fast_quotient)
  {
    int error_id = atomicFetchAdd(results[0], 1);
    if (error_id == 0)
    {
      results[1] = divident;
      results[2] = quotient;
      results[3] = fast_quotient;
    }
  }

  divident = -divident;
  quotient = divident / (int)divisor;
  fast_quotient = divident / divisor;

  if (quotient != fast_quotient)
  {
    int error_id = atomicFetchAdd(results[0], 1);
    if (error_id == 0)
    {
      results[1] = divident;
      results[2] = quotient;
      results[3] = fast_quotient;
    }
  }
}
