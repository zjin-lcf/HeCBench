#include <sycl/sycl.hpp>

template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline T atomicAdd(T* val, const T delta)
{
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     MemoryScope> ref(*val);
    return ref.fetch_add(delta);
}

void kernel1 (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d,
    const sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, sycl::native::exp(sum));
  }
}

void kernel2 (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n,
    const sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < n)
    score[i] = sycl::native::exp(dot_product[i]) / exp_sum[0];
}


void kernel3 (
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d,
    const sycl::nd_item<1> &item)
{
  int j = item.get_global_id(0);
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}


void kernel1_blockReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d,
    const sycl::nd_item<1> &item)
{
  // each i iteration is assigned to a block
  int i = item.get_group(0);
  float sum = 0;
  for (int j = item.get_local_id(0); j < d;
       j += item.get_local_range(0)) {
    sum += key[i * d + j] * query[j];
  }
  sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>{});

  if (item.get_local_id(0) == 0) {
    dot_product[i] = sum;
    atomicAdd(exp_sum, sycl::native::exp(sum));
  }
}


void kernel1_warpReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d,
    const sycl::nd_item<1> &item)
{

  sycl::sub_group warp = item.get_sub_group();
  // each i iteration is assigned to a warp
  int i = item.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
  if (i < n) {
    float sum = 0;
    for (int j = warp.get_local_linear_id(); j < d; j += warp.get_max_local_range()[0]) {
      sum += key[i * d + j] * query[j];
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    if (warp.leader()) {
      dot_product[i] = sum;
      atomicAdd(exp_sum, sycl::native::exp(sum));
    }
  }
}


void kernel2_blockReduce (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d,
    const sycl::nd_item<1> &item)
{
  int j = item.get_group(0);
  float sum = 0;
  for (int i = item.get_local_id(0); i < n;
       i += item.get_local_range(0)) {
    float score = sycl::native::exp(dot_product[i]) / exp_sum[0];
    sum += score * value[i * d + j];
  }

  sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>{});
  if (item.get_local_id(0) == 0)
    output[j] = sum;
}
