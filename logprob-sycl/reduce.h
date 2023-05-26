template<typename T>
inline T warpReduceSum(T val, sycl::nd_item<2> &item)
{
  auto sg = item.get_sub_group();
  #pragma unroll
  for (int mask = SG/2; mask > 0; mask >>= 1)
    val += sycl::permute_group_by_xor(sg, val, mask);
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
inline T blockReduceSum(T val, sycl::nd_item<2> &item, T *shared)
{
#ifdef WAVE64
  int lane = item.get_local_id(1) & 0x3f; // in-warp idx
  int wid = item.get_local_id(1) >> 6;    // warp idx
#else
  int lane = item.get_local_id(1) & 0x1f; // in-warp idx
  int wid = item.get_local_id(1) >> 5;    // warp idx
#endif

  val = warpReduceSum<T>(val, item);

  if (lane == 0)
    shared[wid] = val;

  item.barrier(sycl::access::fence_space::local_space);

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (item.get_local_id(1) < (item.get_local_range(1) / (float)SG))
            ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val, item);

  return val;
}

template<typename T>
inline T warpReduceMax(T val, sycl::nd_item<2> &item)
{
  auto sg = item.get_sub_group();
  #pragma unroll
  for (int mask = SG/2; mask > 0; mask >>= 1)
    val = sycl::max(val, sycl::permute_group_by_xor(sg, val, mask));
  return val;
}

/* Calculate the maximum of all elements in a block */
template<typename T>
inline T blockReduceMax(T val, sycl::nd_item<2> &item, T *shared)
{

#ifdef WAVE64
  int lane = item.get_local_id(1) & 0x3f; // in-warp idx
  int wid = item.get_local_id(1) >> 6;    // warp idx
#else
  int lane = item.get_local_id(1) & 0x1f; // in-warp idx
  int wid = item.get_local_id(1) >> 5;    // warp idx
#endif

  val = warpReduceMax(val, item); // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  item.barrier(sycl::access::fence_space::local_space);

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32

  val = (item.get_local_id(1) < (item.get_local_range(1) / (float)SG))
            ? shared[lane]
            : -1e20f;
  val = warpReduceMax(val, item);

  return val;
}
