/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


static uint iSnapUp(const uint dividend, const uint divisor)
{
  return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
}
uint factorRadix2(uint& log2L, uint L)
{
  if(!L)
  {
    log2L = 0;
    return 0;
  } else {
    for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
    return L;
  }
}


////////////////////////////////////////////////////////////////////////////////
// Scan codelets
////////////////////////////////////////////////////////////////////////////////
#if(1)
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline uint scan1Inclusive(sycl::nd_item<1> &item, const uint idata,
                           uint *l_Data, const uint size)
{
  uint pos = 2 * item.get_local_id(0) - (item.get_local_id(0) & (size - 1));
  l_Data[pos] = 0;
  pos += size;
  l_Data[pos] = idata;

  for(uint offset = 1; offset < size; offset <<= 1){
    item.barrier(sycl::access::fence_space::local_space);
    uint t = l_Data[pos] + l_Data[pos - offset];
    item.barrier(sycl::access::fence_space::local_space);
    l_Data[pos] = t;
  }

  return l_Data[pos];
}

inline uint scan1Exclusive(sycl::nd_item<1> &item, const uint idata,
                           uint *l_Data, const uint size)
{
  return scan1Inclusive(item, idata, l_Data, size) - idata;
}

#else
#define LOG2_WARP_SIZE 5U
#define      WARP_SIZE (1U << LOG2_WARP_SIZE)

//Almost the same as naiveScan1 but doesn't need barriers
//assuming size <= WARP_SIZE
inline uint warpScanInclusive(sycl::nd_item<1> &item, const uint idata,
                              uint *l_Data, const uint size)
{
  uint pos = 2 * item.get_local_id(0) - (item.get_local_id(0) & (size - 1));
  l_Data[pos] = 0;
  pos += size;
  l_Data[pos] = idata;

  for(uint offset = 1; offset < size; offset <<= 1)
    l_Data[pos] += l_Data[pos - offset];

  return l_Data[pos];
}

inline uint warpScanExclusive(sycl::nd_item<1> &item, const uint idata,
                              uint *l_Data, const uint size)
{
  return warpScanInclusive(item, idata, l_Data, size) - idata;
}

inline uint scan1Inclusive(sycl::nd_item<1> &item, const uint idata,
                           uint *l_Data, const uint size)
{
  if(size > WARP_SIZE){
    //Bottom-level inclusive warp scan
    uint warpResult = warpScanInclusive(item, idata, l_Data, WARP_SIZE);

    //Save top elements of each warp for exclusive warp scan
    //sync to wait for warp scans to complete (because l_Data is being overwritten)
    item.barrier(sycl::access::fence_space::local_space);

    int lid = item.get_local_id(0);
    if( (lid & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
      l_Data[lid >> LOG2_WARP_SIZE] = warpResult;

    //wait for warp scans to complete
    item.barrier(sycl::access::fence_space::local_space);
    if( lid < (WORKGROUP_SIZE / WARP_SIZE) ){
      //grab top warp elements
      uint val = l_Data[lid] ;
      //calculate exclsive scan and write back to shared memory
      l_Data[lid] = warpScanExclusive(item, val, l_Data, size >> LOG2_WARP_SIZE);
    }

    //return updated warp scans with exclusive scan results
    item.barrier(sycl::access::fence_space::local_space);
    return warpResult + l_Data[lid >> LOG2_WARP_SIZE];
  }else{
    return warpScanInclusive(item, idata, l_Data, size);
  }
}

inline uint scan1Exclusive(sycl::nd_item<1> &item, const uint idata,
                           uint *l_Data, const uint size){
  return scan1Inclusive(item, idata, l_Data, size) - idata;
}
#endif


//Vector scan: the array to be scanned is stored
//in work-item private memory as uint4
inline uint4 scan4Inclusive(sycl::nd_item<1> &item, uint4 data4,
                            uint *l_Data, const uint size){
  //Level-0 inclusive scan
  data4.y() += data4.x();
  data4.z() += data4.y();
  data4.w() += data4.z();

  //Level-1 exclusive scan
  uint val = scan1Inclusive(item, data4.w(), l_Data, size / 4) - data4.w();

  return (data4 + (uint4)val);
}

inline uint4 scan4Exclusive(sycl::nd_item<1> &item, const uint4 data4,
                            uint *l_Data, const uint size)
{
  return scan4Inclusive(item, data4, l_Data, size) - data4;
}


////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
void scanExclusiveLocal1K(
      sycl::nd_item<1> &item,
      uint *d_Dst,
      uint *d_Src,
      uint *l_Data,
      const uint size)
{
    int i = item.get_global_id(0);

    //Load data
    uint4 idata4 = reinterpret_cast<const uint4*>(d_Src)[i];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(item, idata4, l_Data, size);

    //Write back
    reinterpret_cast<uint4*>(d_Dst)[i] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
void scanExclusiveLocal2K(
      sycl::nd_item<1> &item,
      uint *d_Buf,
      uint *d_Dst,
      uint *d_Src,
      uint *l_Data,
      const uint N,
      const uint arrayLength)
{
    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    //Skip loads and stores for inactive work-items of the work-group with highest index(pos >= N)
    uint data = 0;
    int i = item.get_global_id(0);
    if(i < N)
      data = d_Dst[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * i] +
             d_Src[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * i];

    //Compute
    uint odata = scan1Exclusive(item, data, l_Data, arrayLength);

    //Avoid out-of-bound access
    if(i < N) d_Buf[i] = odata;
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
void uniformUpdateK(
      // uint4 *d_Data,
      sycl::nd_item<1> &item,
      uint *d_Data,
      uint *d_Buf,
      uint &buf)
{
    //__local uint buf[1];

    int i = item.get_global_id(0);
    //uint4 data4 = d_Data[get_global_id(0)];
    uint4 data4 = reinterpret_cast<uint4*>(d_Data)[i];

    if(item.get_local_id(0) == 0)
      buf = d_Buf[item.get_group(0)];

    item.barrier(sycl::access::fence_space::local_space);
    data4 += (uint4)buf;

    //d_Data[i] = data4;
    reinterpret_cast<uint4*>(d_Data)[i] = data4;
}
