static const float HALF_FLT_MAX = 65504.F;

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
  #ifdef WAVE64
  for (int mask = 32; mask > 0; mask >>= 1)
    val += __shfl_xor(val, mask);
  #else
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor(val, mask, 32);
  #endif
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
  #ifdef WAVE64
  static __shared__ T shared[64];
  int lane = threadIdx.x & 0x3f;
  int wid = threadIdx.x >> 6;
  #else
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  #endif

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  #ifdef WAVE64
  val = (threadIdx.x < (blockDim.x / 64.f)) ? shared[lane] : (T)(0.0f);
  #else
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  #endif
  val = warpReduceSum<T>(val);

  return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
  #pragma unroll
  #ifdef WAVE64
  for (int mask = 32; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor(val, mask));
  #else
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor(val, mask, 32));
  #endif
  return val;
}

/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
  #ifdef WAVE64
  static __shared__ T shared[64];
  int lane = threadIdx.x & 0x3f;  // in-warp idx
  int wid = threadIdx.x >> 6;     // warp idx
  #else
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx
  #endif

  val = warpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  #ifdef WAVE64
  val = (threadIdx.x < (blockDim.x / 64.f)) ? shared[lane] : -1e20f;
  #else
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  #endif
  val = warpReduceMax(val);

  return val;
}
