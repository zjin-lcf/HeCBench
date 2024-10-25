#define NUM_OF_BLOCKS (1024 * 1024)
#define NUM_OF_THREADS 128

inline
void reduceInShared_native(sycl::half2 *const v, sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = v[lid] + v[lid+i];
    item.barrier(sycl::access::fence_space::local_space);
  }
}

void scalarProductKernel_native(const sycl::half2 *a,
                                const sycl::half2 *b,
                                float *results,
                                      sycl::half2 *shArray,
                                const size_t size,
                                sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  int gid = item.get_group(0);

  const int stride = item.get_group_range(0) * item.get_local_range(0);

  sycl::half2 value(0.f, 0.f);
  shArray[lid] = value;

  for (int i = item.get_global_id(0); i < size; i += stride)
  {
    value = a[i] * b[i] + value;
  }

  shArray[lid] = value;
  item.barrier(sycl::access::fence_space::local_space);
  reduceInShared_native(shArray, item);

  if (lid == 0)
  {
    sycl::half2 result = shArray[0];
    float f_result = (float)result.y() + (float)result.x();
    auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                     sycl::memory_scope::device,\
                     sycl::access::address_space::global_space>(results[0]);
    ao.fetch_add(f_result);
  }
}

void scalarProductKernel_native2(const sycl::half2 *a,
                                 const sycl::half2 *b,
                                 float *results,
                                 const size_t size,
                                 sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  int gid = item.get_group(0);

  const int stride = item.get_group_range(0) * item.get_local_range(0);

  sycl::half2 value(0.f, 0.f);

  for (int i = item.get_global_id(0); i < size; i += stride)
  {
    value += a[i] * b[i];
  }

  value = sycl::reduce_over_group(item.get_group(), value, sycl::plus<sycl::half2>{});

  if (lid == 0)
  {
    float f_result = (float)value.y() + (float)value.x();
    auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                     sycl::memory_scope::device,\
                     sycl::access::address_space::global_space>(results[0]);
    ao.fetch_add(f_result);
  }
}

void scalarProductKernel_native3(const sycl::float4 *a,
                                 const sycl::float4 *b,
                                 float *results,
                                 const size_t size,
                                 sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  int gid = item.get_group(0);

  const int stride = item.get_group_range(0) * item.get_local_range(0);

  sycl::half2 value(0.f, 0.f);

  for (int i = item.get_global_id(0); i < size/4; i += stride)
  {
    const sycl::half2 *a_v = reinterpret_cast<const sycl::half2*>(&a[i]);
    const sycl::half2 *b_v = reinterpret_cast<const sycl::half2*>(&b[i]);
    value += a_v[0] * b_v[0];
    value += a_v[1] * b_v[1];
    value += a_v[2] * b_v[2];
    value += a_v[3] * b_v[3];
  }

  value = sycl::reduce_over_group(item.get_group(), value, sycl::plus<sycl::half2>{});

  if (lid == 0)
  {
    float f_result = (float)value.y() + (float)value.x();
    auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                     sycl::memory_scope::device,\
                     sycl::access::address_space::global_space>(results[0]);
    ao.fetch_add(f_result);
  }
}


