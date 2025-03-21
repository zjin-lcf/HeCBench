#include <sycl/sycl.hpp>
#include "atomics.h"

void attention_kernel1 (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int i = item.get_global_id(2);
      if (i < n) {
        float sum = 0;
        for (int j = 0; j < d; j++)
          sum += key[i * d + j] * query[j];
        dot_product[i] = sum;
        atomicAdd(exp_sum[0], sycl::native::exp(sum));
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void attention_kernel2 (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int i = item.get_global_id(2);
      if (i < n)
        score[i] = sycl::native::exp(dot_product[i]) / exp_sum[0];
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}


void attention_kernel3 (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int j = item.get_global_id(2);
      if (j < d) {
        float sum = 0;
        for (int i = 0; i < n; i++)
          sum += score[i] * value[i * d + j];
        output[j] = sum;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}


void attention_kernel1_blockReduce (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      // each i iteration is assigned to a block
      int i = item.get_group(2);
      float sum = 0;
      for (int j = item.get_local_id(2); j < d;
           j += item.get_local_range(2)) {
        sum += key[i * d + j] * query[j];
      }
      sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>{});

      if (item.get_local_id(2) == 0) {
        dot_product[i] = sum;
        atomicAdd(exp_sum[0], sycl::native::exp(sum));
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}


void attention_kernel1_warpReduce (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {

      sycl::sub_group warp = item.get_sub_group();
      // each i iteration is assigned to a warp
      int i = item.get_group(2) * warp.get_group_linear_range() + warp.get_group_linear_id();
      if (i < n) {
        float sum = 0;
        for (int j = warp.get_local_linear_id(); j < d; j += warp.get_max_local_range()[0]) {
          sum += key[i * d + j] * query[j];
        }
        sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
        if (warp.leader()) {
          dot_product[i] = sum;
          atomicAdd(exp_sum[0], sycl::native::exp(sum));
        }
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}


void attention_kernel2_blockReduce (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
  int j = item.get_group(2);
  float sum = 0;
  for (int i = item.get_local_id(2); i < n;
       i += item.get_local_range(2)) {
    float score = sycl::native::exp(dot_product[i]) / exp_sum[0];
    sum += score * value[i * d + j];
  }

  sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>{});
  if (item.get_local_id(2) == 0)
    output[j] = sum;
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void attention_kernel2_warpReduce (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
  sycl::sub_group warp = item.get_sub_group();
  int j = item.get_group(2) * warp.get_group_linear_range() + warp.get_group_linear_id();
  if (j < d) {
    float sum = 0;
      for (int i = warp.get_local_linear_id(); i < n; i += warp.get_max_local_range()[0]) {
      float score = sycl::native::exp(dot_product[i]) / exp_sum[0];
      sum += score * value[i * d + j];
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    if (warp.leader())
      output[j] = sum;
  }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
