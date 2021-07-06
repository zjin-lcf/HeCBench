//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef _REDUCTION_H_
#define _REDUCTION_H_

/* warp reduction with shfl function */
template < typename T >
inline float warp_reduce(T val, nd_item<3> &item)
{
  for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
    val += item.get_sub_group().shuffle_down(val, offset);
  return val;
}

/* block reduction with warp reduction */
template < typename T >
inline float block_reduce(T val, T* shared, nd_item<3> &item)
{
  int lx = item.get_local_id(2);
  int ly = item.get_local_id(1);
  int lz = item.get_local_id(0);
  int bx = item.get_local_range(2); 
  int by = item.get_local_range(1); 
  int bz = item.get_local_range(0); 
  int tid = lz * BY * BX + ly * BX + lx;
  int lane = tid & (WARPSIZE-1);
  int wid = tid/WARPSIZE;
  val = warp_reduce<T>(val, item);

  if(lane == 0) shared[wid] = val;

  item.barrier(access::fence_space::local_space);

  val = (tid < (bx * by * bz)/WARPSIZE) ? shared[lane] : 0;
  if(wid == 0) val = warp_reduce<T>(val, item);
  return val;
}


/* energy reduction using block reduction */
template <typename T>
void redenergy(queue &q, 
               range<3> gws,
               range<3> lws,
               buffer<int> &mdlat, 
               int L,
               buffer<T, 1> dE,
               const int k,      // dE + k
               buffer<int> dH,
               float h)
{
  
  q.submit([&] (handler &cgh) {
    auto s = mdlat.get_access<sycl_read>(cgh);
    auto out = dE.template get_access<sycl_read_write>(cgh);
    auto H = dH.get_access<sycl_read>(cgh);
    accessor<T, 1, sycl_read_write, access::target::local> lmem (WARPSIZE, cgh);
    cgh.parallel_for<class setup_pcg>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      int x = item.get_global_id(2);
      int y = item.get_global_id(1);
      int z = item.get_global_id(0);
      int lx = item.get_local_id(2);
      int ly = item.get_local_id(1);
      int lz = item.get_local_id(0);
      int tid = lz * BY * BX + ly * BX + lx;
      int id = C(x,y,z,L);
      // this optimization only works for L being a power of 2
      //float sum = -(float)(s[id] * ((float)(s[C((x+1) & (L-1), y, z, L)] + 
      // s[C(x, (y+1) & (L-1), z, L)] + s[C(x, y, (z+1) & (L-1), L)]) + h*H[id]));

      // this line works always
      float sum = -(float)(s[id] * ((float)(s[C((x+1) >=  L? 0: x+1, y, z, L)] + 
                  s[C(x, (y+1) >= L? 0 : y+1, z, L)] + s[C(x, y, (z+1) >= L? 0 : z+1, L)]) + h*H[id]));
      sum = block_reduce<T>(sum, lmem.get_pointer(), item); 

      if(tid == 0) {
        auto atomic_obj_ref = ONEAPI::atomic_ref<float,
                     ONEAPI::memory_order::relaxed, 
                     ONEAPI::memory_scope::device, 
                     access::address_space::global_space> (out[k]);
        atomic_obj_ref.fetch_add(sum);
      }
    });
  });
}


#endif
