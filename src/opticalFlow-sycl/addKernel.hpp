#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
///
/// CUDA kernel
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
void AddKernel(const float *op1, const float *op2, int count,
                          float *sum, const sycl::nd_item<3> &item) {
  const int pos = item.get_local_id(2) +
                  item.get_group(2) * item.get_local_range(2);

  if (pos >= count) return;

  sum[pos] = op1[pos] + op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
static void Add(const float *op1, const float *op2, int count, float *sum, sycl::queue &q) {
  sycl::range<3> threads(1, 1, 256);
  sycl::range<3> blocks(1, 1, iDivUp(count, threads[2]));

  q.parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item) {
        AddKernel(op1, op2, count, sum, item);
      });
}
