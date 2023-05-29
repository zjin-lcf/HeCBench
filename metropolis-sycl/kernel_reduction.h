//  trueke // A multi-GPU implementation of the exchange Monte Carlo method. //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang. //
//                                                                              //
//  This file is part of trueke. // trueke is free software: you can
//  redistribute it and/or modify              // it under the terms of the GNU
//  General Public License as published by        // the Free Software
//  Foundation, either version 3 of the License, or           // (at your
//  option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful, // but WITHOUT ANY
//  WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the // GNU General
//  Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License // along
//  with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef _REDUCTION_H_
#define _REDUCTION_H_

/* warp reduction with shfl function */
template < typename T >
inline float warp_reduce(T val, sycl::nd_item<3> &item)
{
  auto sg = item.get_sub_group();
  for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
    val += sg.shuffle_down(val, offset);
  return val;
}

/* block reduction with warp reduction */
template < typename T >
inline float block_reduce(T val, sycl::nd_item<3> &item, T *shared)
{
  int tid = item.get_local_id(0) * BY * BX + item.get_local_id(1) * BX +
            item.get_local_id(2);
  int lane = tid & (WARPSIZE-1);
  int wid = tid/WARPSIZE;
  val = warp_reduce<T>(val, item);

  if(lane == 0)
    shared[wid] = val;

  item.barrier(sycl::access::fence_space::local_space);

  val = (tid < (item.get_local_range(2) * item.get_local_range(1) *
                item.get_local_range(0)) /
                   WARPSIZE)
            ? shared[lane]
            : 0;
  if(wid == 0){
    val = warp_reduce<T>(val, item);
  }
  return val;
}


/* energy reduction using block reduction */
template <typename T>
void kernel_redenergy(const int *s, int L, T *out, const int *H, float h,
                      sycl::nd_item<3> &item, T *shared)
{
  // offsets
  int x = item.get_group(2) * item.get_local_range(2) +
          item.get_local_id(2);
  int y = item.get_group(1) * item.get_local_range(1) +
          item.get_local_id(1);
  int z = item.get_group(0) * item.get_local_range(0) +
          item.get_local_id(0);
  int tid = item.get_local_id(0) * BY * BX + item.get_local_id(1) * BX +
            item.get_local_id(2);
  int id = C(x,y,z,L);
  // this optimization only works for L being a power of 2
  //float sum = -(float)(s[id] * ((float)(s[C((x+1) & (L-1), y, z, L)] +
  // s[C(x, (y+1) & (L-1), z, L)] + s[C(x, y, (z+1) & (L-1), L)]) + h*H[id]));

  // this line works always
  float sum = -(float)(s[id] * ((float)(s[C((x+1) >=  L? 0: x+1, y, z, L)] +
              s[C(x, (y+1) >= L? 0 : y+1, z, L)] + s[C(x, y, (z+1) >= L? 0 : z+1, L)]) + h*H[id]));
  sum = block_reduce<T>(sum, item, shared);

  if (tid == 0) {
    auto ao = sycl::atomic_ref<T,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (out[0]);
     ao.fetch_add(sum);
  }
}

#endif
