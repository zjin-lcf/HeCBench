#include <hipcub/hipcub.hpp>
#include <hip/hip_cooperative_groups.h>

__global__
void kernel1 (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, expf(sum));
  }
}

__global__
void kernel2 (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    score[i] = expf(dot_product[i]) / exp_sum[0];
}

__global__
void kernel3 (
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}

__global__
void kernel1_blockReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  // each i iteration is assigned to a block
  int i = blockIdx.x;
  float sum = 0;
  for (int j = threadIdx.x; j < d; j += blockDim.x) {
    sum += key[i * d + j] * query[j];
  }
  using BlockReduce = hipcub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  sum = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    dot_product[i] = sum;
    atomicAdd(exp_sum, expf(sum));
  }
}

__device__ inline float warpReduceSum(cooperative_groups::thread_block_tile<32> &warp, float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp.shfl_xor(val, offset);
    }
    return val;
}

__global__
void kernel1_warpReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  // each i iteration is assigned to a warp
  // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i < n) {
    float sum = 0;
    for (int j = warp.thread_rank(); j < d; j += warp.size()) {
      sum += key[i * d + j] * query[j];
    }
    //sum = cg::reduce(warp, sum, cg::plus<float>{});
    sum = warpReduceSum(warp, sum);
    if (warp.thread_rank() == 0) {
      dot_product[i] = sum;
      atomicAdd(exp_sum, expf(sum));
    }
  }
}

__global__
void kernel2_blockReduce (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  int j = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float score = expf(dot_product[i]) / exp_sum[0];
    sum += score * value[i * d + j];
  }
  using BlockReduce = hipcub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  sum = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0)
    output[j] = sum;
}
