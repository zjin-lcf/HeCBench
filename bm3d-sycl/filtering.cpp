#include <float.h>
#include <stdio.h>
#include "common.h"

#include "indices.hpp"
#include "params.hpp"


// Kernels used for collaborative filtering and aggregation

//Sum the passed values in a warp to the first thread of this warp.
template<typename T>
inline T warpReduceSum(nd_item<2> &item, T val) 
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += item.get_sub_group().shuffle_down(val, offset);
  return val;
}


//Sum the passed values in a block to the first thread of a block.
template<typename T>
inline float blockReduceSum(nd_item<2> &item, T* shared, T val, int tid, int tcount) 
{
  int lane = tid % warpSize;
  int wid = tid / warpSize;

  val = warpReduceSum(item, val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  item.barrier(access::fence_space::local_space);              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (tid < tcount / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(item, val); //Final reduce within first warp

  return val;
}

//Returns absolute value of the passed real number raised to the power of two
inline float abspow2(float & a)
{
  return a * a;
}


//Integer logarithm base 2.
template <typename IntType>
inline uint ilog2(IntType n)
{
  uint l;
  for (l = 0; n; n >>= 1, ++l);
  return l;
}


//Orthogonal transformation.
template <typename T>
inline void rotate(T& a, T& b)
{
  T tmp;
  tmp = a;
  a = tmp + b;
  b = tmp - b;
}


//Fast Walsh-Hadamard transform.
template <typename T>
inline void fwht(T *data, uint n)
{
  unsigned l2 = ilog2(n) - 1;
  for ( uint i = 0; i < l2; ++i )
  {
    for (uint j = 0; j < n; j += (1 << (i + 1)))
    for (uint k = 0; k < (uint)(1 << i); ++k)
      rotate(data[j + k], data[j + k + (uint)(1 << i)]);
  }
}

//Based on blockIdx it computes the addresses to the arrays in global memory
inline void get_block_addresses(
  nd_item<2> &item,
  const uint2 & start_point,    //IN: first reference patch of a batch
  const uint & patch_stack_size,  //IN: maximal size of a 3D group
  const uint2 & stacks_dim,    //IN: Size of area, where reference patches could be located
  const Params & params,      //IN: Denoising parameters
  uint2 & outer_address,      //OUT: Coordinetes of reference patch in the image
  uint & start_idx)        //OUT: Address of a first element of the 3D group in stacks array
{
  const int bidx = item.get_group(1);
  const int bidy = item.get_group(0);
  const int gridx = item.get_group_range(1);
  //One block handles one patch_stack, data are in array one after one.
  start_idx = patch_stack_size * idx2(bidx, bidy, gridx);
  
  outer_address.x() = start_point.x() + (bidx * params.p);
  outer_address.y() = start_point.y() + (bidy * params.p);

  //Ensure, that the bottom most patches will be taken as reference patches regardless the p parameter.
  if (outer_address.y() >= stacks_dim.y() && outer_address.y() < stacks_dim.y() + params.p - 1)
    outer_address.y() = stacks_dim.y() - 1;
  //Ensure, that the right most patches will be taken as reference patches regardless the p parameter.
  if (outer_address.x() >= stacks_dim.x() && outer_address.x() < stacks_dim.x() + params.p - 1)
    outer_address.x() = stacks_dim.x() - 1;
}

/*
Gather patches form image based on matching stored in 3D array stacks
Used parameters: p,k,N
Division: One block handles one patch_stack, threads match to the pixels of a patch
*/
void get_block(
    nd_item<2> &item,
    const uint2 start_point,         //IN: first reference patch of a batch
    const uchar* __restrict image,        //IN: image
    const ushort* __restrict stacks,        //IN: array of adresses of similar patches
    const uint* __restrict g_num_patches_in_stack,    //IN: numbers of patches in 3D groups
    float* patch_stack,          //OUT: assembled 3D groups
    const uint2 image_dim,          //IN: image dimensions
    const uint2 stacks_dim,          //IN: dimensions limiting addresses of reference patches
    const Params params)           //IN: denoising parameters
{
  uint startidx;
  uint2 outer_address;
  get_block_addresses(item, start_point,  params.k*params.k*(params.N+1), stacks_dim, params, outer_address, startidx);

  if (outer_address.x() >= stacks_dim.x() || outer_address.y() >= stacks_dim.y()) return;
  
  const int lidx = item.get_local_id(1);
  const int lidy = item.get_local_id(0);
  const int bidx = item.get_group(1);
  const int bidy = item.get_group(0);
  const int gridx = item.get_group_range(1);

  patch_stack += startidx;
  
  const ushort* z_ptr = &stacks[ idx3(0, bidx, bidy, params.N,  gridx) ];

  uint num_patches = g_num_patches_in_stack[ idx2(bidx, bidy, gridx) ];
  
  patch_stack[ idx3(lidx, lidy, 0, params.k, params.k) ] = 
    (float)(image[ idx2(outer_address.x()+lidx, outer_address.y()+lidy, image_dim.x())]);
  for(uint i = 0; i < num_patches; ++i)
  {
    int x = (int)((signed char)(z_ptr[i] & 0xFF));
    int y = (int)((signed char)((z_ptr[i] >> 8) & 0xFF));
    patch_stack[ idx3(lidx, lidy, i+1, params.k, params.k) ] = 
      (float)(image[ idx2(outer_address.x()+x+lidx, outer_address.y()+y+lidy, image_dim.x())]);
  }
}

/*
1) Do the Walsh-Hadamard 1D transform on the z axis of 3D stack. 
2) Treshold every pixel and count the number of non-zero coefficients
3) Do the inverse Walsh-Hadamard 1D transform on the z axis of 3D stack.
Used parameters: L3D,N,k,p
Division: Each block delas with one transformed patch stack. (number of threads in block should be k*k)
*/
void hard_treshold_block(
  nd_item<2> &item,
  float *__restrict data,
  const uint2 start_point,    //IN: first reference patch of a batch
  float* __restrict patch_stack,        //IN/OUT: 3D groups with thransfomed patches
  float* __restrict w_P,          //OUT: weight of each 3D group
  const uint* __restrict g_num_patches_in_stack,  //IN: numbers of patches in 3D groups
  uint2 stacks_dim,        //IN: dimensions limiting addresses of reference patches
  const Params params,      //IN: denoising parameters
  const uint sigma        //IN: noise variance
)
{

  const int lidx = item.get_local_id(1);
  const int lidy = item.get_local_id(0);
  const int dimx = item.get_local_range(1);
  const int dimy = item.get_local_range(0);
  const int bidx = item.get_group(1);
  const int bidy = item.get_group(0);
  const int gridx = item.get_group_range(1);

  int paramN = params.N+1;
  uint tcount = dimx*dimy;
  uint tid = idx2(lidx, lidy, dimx);
  uint patch_stack_size = tcount * paramN;

  uint startidx;
  uint2 outer_address;
  get_block_addresses(item, start_point, patch_stack_size, stacks_dim, params, outer_address, startidx);
  
  if (outer_address.x() >= stacks_dim.x() || outer_address.y() >= stacks_dim.y()) return;

  uint num_patches = g_num_patches_in_stack[ idx2(bidx, bidy, gridx) ]+1; //+1 for the reference patch.
  float* s_patch_stack = data + (tid * (num_patches+1)); //+1 for avoiding bank conflicts //TODO:sometimes
  patch_stack = patch_stack + startidx + tid;
    
  //Load to the shared memory
  for(uint i = 0; i < num_patches; ++i)
    s_patch_stack[i] = patch_stack[ i*tcount ];  

  //1D Transform
  fwht(s_patch_stack, num_patches);
  
  //Hard-thresholding + counting of nonzero coefficients
  uint nonzero = 0;
  float threshold = params.L3D * sycl::sqrt((float)(num_patches * sigma));
  for(int i = 0; i < num_patches; ++i)
  {
    if (sycl::fabs(s_patch_stack[ i ]) < threshold)
    {
      s_patch_stack[ i ] = 0.0f;
    }
    else 
      ++nonzero;
  }
  
  //Inverse 1D Transform
  fwht(s_patch_stack, num_patches);
  
  //Normalize and save to global memory
  for (uint i = 0; i < num_patches; ++i)
  {
    patch_stack[ i*tcount ] = s_patch_stack[i] / num_patches;
  }
  
  //Reuse the shared memory for 32 partial sums
  item.barrier(access::fence_space::local_space);
  uint* shared = (uint*)data;
  //Sum the number of non-zero coefficients for a 3D group
  nonzero = blockReduceSum<uint>(item, shared, nonzero, tid, tcount);
  
  //Save the weight of a 3D group (1/nonzero coefficients)
  if (tid == 0)
  {
    if (nonzero < 1) nonzero = 1;
    w_P[ idx2(bidx, bidy, gridx ) ] = 1.0f/(float)nonzero;
  }
}

/*
Fills two buffers: numerator and denominator in order to compute weighted average of pixels
Used parameters: k,N,p
Division: Each block delas with one transformed patch stack.
*/
void aggregate_block(
  nd_item<2> &item,
  const uint2 start_point,      //IN: first reference patch of a batch
  const float* __restrict patch_stack,    //IN: 3D groups with thransfomed patches
  const float* __restrict w_P,      //IN: weight for each 3D group
  const ushort* __restrict stacks,      //IN: array of adresses of similar patches
  const float* __restrict kaiser_window,    //IN: kaiser window
  float* numerator,        //IN/OUT: numerator aggregation buffer (have to be initialized to 0)
  float* denominator,        //IN/OUT: denominator aggregation buffer (have to be initialized to 0)
  const uint* __restrict g_num_patches_in_stack,  //IN: numbers of patches in 3D groups
  const uint2 image_dim,        //IN: image dimensions
  const uint2 stacks_dim,        //IN: dimensions limiting addresses of reference patches
  const Params params        //IN: denoising parameters
)
{    
  const int lidx = item.get_local_id(1);
  const int lidy = item.get_local_id(0);
  const int bidx = item.get_group(1);
  const int bidy = item.get_group(0);
  //const int dimx = item.get_local_range(1);
  const int gridx = item.get_group_range(1);

  uint startidx;
  uint2 outer_address;
  get_block_addresses(item, start_point, params.k*params.k*(params.N+1), stacks_dim, params, outer_address, startidx);
  
  if (outer_address.x() >= stacks_dim.x() || outer_address.y() >= stacks_dim.y()) return;

  patch_stack += startidx;

  uint num_patches = g_num_patches_in_stack[ idx2(bidx, bidy, gridx) ]+1;

  float wp = w_P[ idx2(bidx, bidy, gridx ) ];
  
  const ushort* z_ptr = &stacks[ idx3(0, bidx, bidy, params.N,  gridx) ];

  float kaiser_value = kaiser_window[ idx2(lidx, lidy, params.k) ];

  for(uint z = 0; z < num_patches; ++z)
  {
    int x = 0;
    int y = 0;
    if (z > 0) {
      x = (int)((signed char)(z_ptr[z-1] & 0xFF));
      y = (int)((signed char)((z_ptr[z-1] >> 8) & 0xFF));
    }

    float value = ( patch_stack[ idx3(lidx, lidy, z, params.k, params.k) ]);
    int idx = idx2(outer_address.x() + x + lidx, outer_address.y() + y + lidy, image_dim.x());

    auto num_ref = ext::oneapi::atomic_ref<float,
                   ext::oneapi::memory_order::relaxed, 
                   ext::oneapi::memory_scope::device, 
                   access::address_space::global_space> (numerator[idx]);
    num_ref.fetch_add(value * kaiser_value * wp);

    auto den_ref = ext::oneapi::atomic_ref<float,
                   ext::oneapi::memory_order::relaxed, 
                   ext::oneapi::memory_scope::device, 
                   access::address_space::global_space> (denominator[idx]);
    den_ref.fetch_add(kaiser_value * wp);
  }
}

/*
Divide numerator with denominator and round result to image_o
*/
void aggregate_final(
  nd_item<2> &item,
  const float* __restrict numerator,  //IN: numerator aggregation buffer
  const float* __restrict denominator,  //IN: denominator aggregation buffer
  const uint2 image_dim,      //IN: image dimensions
  uchar*__restrict result)        //OUT: image estimate
{
  uint idx = item.get_global_id(1);
  uint idy = item.get_global_id(0);
  if (idx >= image_dim.x() || idy >= image_dim.y()) return;

  int value = sycl::rint(numerator[ idx2(idx,idy,image_dim.x()) ] /
                         denominator[ idx2(idx,idy,image_dim.x()) ]);
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  result[ idx2(idx,idy,image_dim.x()) ] = (uchar)value;
}


extern "C" void run_get_block(
  queue &q,
  const uint2 start_point,
  buffer<uchar, 1> &image,
  buffer<ushort, 1> &stacks,
  buffer<uint, 1> &num_patches_in_stack,
  buffer<float, 1> &patch_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const range<2> lws,
  const range<2> gws)
{
  q.submit([&] (handler &cgh) {
    auto image_acc = image.get_access<sycl_read>(cgh);
    auto stacks_acc = stacks.get_access<sycl_read>(cgh);
    auto num_patches_in_stack_acc = num_patches_in_stack.get_access<sycl_read>(cgh);
    auto patch_stack_acc = patch_stack.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class assemble>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      get_block(
        item,
        start_point,
        image_acc.get_pointer(),
        stacks_acc.get_pointer(),
        num_patches_in_stack_acc.get_pointer(),
        patch_stack_acc.get_pointer(),
        image_dim,
        stacks_dim,
        params
      );
    });
  });
}

extern "C" void run_hard_treshold_block(
  queue &q,
  const uint2 start_point,
  buffer<float, 1> &patch_stack,
  buffer<float, 1> &w_P,
  buffer<uint, 1> &num_patches_in_stack,
  const uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const range<2> lws,  
  const range<2> gws,
  const uint shared_memory_size)
{
  q.submit([&] (handler &cgh) {
    auto patch_stack_acc = patch_stack.get_access<sycl_read_write>(cgh);
    auto w_P_acc = w_P.get_access<sycl_write>(cgh);
    auto num_patches_in_stack_acc = num_patches_in_stack.get_access<sycl_read>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> 
      lmem(shared_memory_size/sizeof(float), cgh);
    cgh.parallel_for<class hard_treshold>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      hard_treshold_block(
        item,
        lmem.get_pointer(),
        start_point,
        patch_stack_acc.get_pointer(),
        w_P_acc.get_pointer(),
        num_patches_in_stack_acc.get_pointer(),
        stacks_dim,
        params,
        sigma
      );
    });
  });
}

extern "C" void run_aggregate_block(
  queue &q,
  const uint2 start_point,
  buffer<float, 1> &patch_stack,  
  buffer<float, 1> &w_P,
  buffer<ushort, 1> &stacks,
  buffer<float, 1> &kaiser_window,
  buffer<float, 1> &numerator,
  buffer<float, 1> &denominator,
  buffer<uint, 1> &num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const range<2> lws,  
  const range<2> gws)
{
  q.submit([&] (handler &cgh) {
    auto patch_stack_acc = patch_stack.get_access<sycl_read>(cgh);
    auto w_P_acc = w_P.get_access<sycl_read>(cgh);
    auto stacks_acc = stacks.get_access<sycl_read>(cgh);
    auto kaiser_window_acc = kaiser_window.get_access<sycl_read>(cgh);
    auto numerator_acc = numerator.get_access<sycl_read_write>(cgh);
    auto denominator_acc = denominator.get_access<sycl_read_write>(cgh);
    auto num_patches_in_stack_acc = num_patches_in_stack.get_access<sycl_read>(cgh);
    cgh.parallel_for<class aggregate>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      aggregate_block(
        item,
        start_point,
        patch_stack_acc.get_pointer(),
        w_P_acc.get_pointer(),
        stacks_acc.get_pointer(),
        kaiser_window_acc.get_pointer(),
        numerator_acc.get_pointer(),
        denominator_acc.get_pointer(),
        num_patches_in_stack_acc.get_pointer(),
        image_dim,
        stacks_dim,
        params
      );
    });
  });
}

extern "C" void run_aggregate_final(
  queue &q,
  buffer<float, 1> &numerator,
  buffer<float, 1> &denominator,
  const uint2 image_dim,
  buffer<uchar, 1> &denoised_image,
  const range<2> lws,  
  const range<2> gws
)
{
  q.submit([&] (handler &cgh) {
    auto numerator_acc = numerator.get_access<sycl_read>(cgh);
    auto denominator_acc = denominator.get_access<sycl_read>(cgh);
    auto denoised_image_acc = denoised_image.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class final>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      aggregate_final(
        item,
        numerator_acc.get_pointer(),
        denominator_acc.get_pointer(),
        image_dim,
        denoised_image_acc.get_pointer()
      );  
    });
  });
}

