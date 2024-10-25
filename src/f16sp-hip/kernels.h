#define NUM_OF_BLOCKS (1024 * 1024)
#define NUM_OF_THREADS 128

__forceinline__ __device__
void reduceInShared_intrinsics(half2 * const v)
{
  int lid = threadIdx.x;
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = __hadd2(v[lid], v[lid+i]);
    __syncthreads();
  }
}

__forceinline__ __device__
void reduceInShared_native(half2 * const v)
{
  int lid = threadIdx.x;
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = v[lid] + v[lid+i];
    __syncthreads();
  }
}

__global__
void scalarProductKernel_intrinsics(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  const int stride = gridDim.x*blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
  {
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_intrinsics(shArray);

  if (threadIdx.x == 0)
  {
    half2 result = shArray[0];
    float f_result = __low2float(result) + __high2float(result);
    unsafeAtomicAdd(results, f_result);
  }
}

__global__
void scalarProductKernel_native(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  const int stride = gridDim.x*blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value(0.f, 0.f);
  shArray[threadIdx.x] = value;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
  {
    value += a[i] * b[i];
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_native(shArray);

  if (threadIdx.x == 0)
  {
    half2 result = shArray[0];
    float f_result = (float)result.y + (float)result.x;
    unsafeAtomicAdd(results, f_result);
  }
}

__global__
void scalarProductKernel_intrinsics2(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  typedef hipcub::BlockReduce<half2, NUM_OF_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int stride = gridDim.x*blockDim.x;

  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
  {
    value = __hfma2(a[i], b[i], value);
  }

  value = BlockReduce(temp_storage).Sum(value);

  if (threadIdx.x == 0)
  {
    float f_result = __low2float(value) + __high2float(value);
    unsafeAtomicAdd(results, f_result);
  }
}

__global__
void scalarProductKernel_intrinsics3(
    float4 const *__restrict__ const a,
    float4 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  typedef hipcub::BlockReduce<half2, NUM_OF_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int stride = gridDim.x*blockDim.x;

  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size/4; i+=stride)
  {
    const half2 *a_v = reinterpret_cast<const half2*>(&a[i]);
    const half2 *b_v = reinterpret_cast<const half2*>(&b[i]);
    value = __hfma2(a_v[0], b_v[0], value);
    value = __hfma2(a_v[1], b_v[1], value);
    value = __hfma2(a_v[2], b_v[2], value);
    value = __hfma2(a_v[3], b_v[3], value);
  }

  value = BlockReduce(temp_storage).Sum(value);

  if (threadIdx.x == 0)
  {
    float f_result = __low2float(value) + __high2float(value);
    unsafeAtomicAdd(results, f_result);
  }
}

