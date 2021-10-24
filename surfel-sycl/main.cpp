/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "common.h"

#define COL_P_X 0
#define COL_P_Y 1
#define COL_P_Z 2
#define COL_N_X 3
#define COL_N_Y 4
#define COL_N_Z 5
#define COL_RSq 6
#define COL_DIM 7

// forward declaration
template<typename T>
class k;

// compute the xyz images using the inverse focal length invF
template<typename T>
void surfel_render(
  nd_item<2> &item,
  const T *__restrict__ s,
  int N,
  T f,
  int w,
  int h,
  T *__restrict__ d)
{
  const int idx = item.get_global_id(1);
  const int idy = item.get_global_id(0);

  if(idx < w && idy < h)
  {
    T ray[3];
    ray[0] = T(idx)-(w-1)*(T)0.5;
    ray[1] = T(idy)-(h-1)*(T)0.5;
    ray[2] = f;
    T pt[3];
    T n[3];
    T p[3];
    T dMin = 1e20;
    
    for (int i=0; i<N; ++i) {
      p[0] = s[i*COL_DIM+COL_P_X];
      p[1] = s[i*COL_DIM+COL_P_Y];
      p[2] = s[i*COL_DIM+COL_P_Z];
      n[0] = s[i*COL_DIM+COL_N_X];
      n[1] = s[i*COL_DIM+COL_N_Y];
      n[2] = s[i*COL_DIM+COL_N_Z];
      T rSqMax = s[i*COL_DIM+COL_RSq];
      T pDotn = p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
      T dsDotRay = ray[0]*n[0] + ray[1]*n[1] + ray[2]*n[2];
      T alpha = pDotn / dsDotRay;
      pt[0] = ray[0]*alpha - p[0];
      pt[1] = ray[1]*alpha - p[1];
      pt[2] = ray[2]*alpha - p[2];
      T t = ray[2]*alpha;
      T rSq = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
      if (rSq < rSqMax && dMin > t) {
        dMin = t; // ray hit the surfel 
      }
    }
    d[id*w+idx] = dMin > (T)100 ? (T)0 : dMin;
  }
}

template <typename T>
void surfelRenderTest(queue &q, int n, int w, int h)
{
  const int src_size = n*7;
  const int dst_size = w*h;

  T *h_dst = (T*) malloc (dst_size * sizeof(T));
  T *h_src = (T*) malloc (src_size * sizeof(T));

  srand(123);
  for (int i = 0; i < src_size; i++)
    h_src[i] = rand() % 256;

  T inverseFocalLength[3] = {0.005, 0.02, 0.036};

  buffer<T, 1> d_src (h_src, src_size);
  buffer<T, 1> d_dst (dst_size);

  range<2> lws (16, 16);
  range<2> gws ((h+15)/16*16, (w+15)/16*16);
  for (int f = 0; f < 3; f++) {
    for (int i = 0; i < 100; i++)
      q.submit([&] (handler &cgh) {
        auto src = d_src.template get_access<sycl_read>(cgh);
        auto dst = d_dst.template get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class k<T>>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          surfel_render<T>(item, src.get_pointer(), n, 
                           inverseFocalLength[f], w, h,
                           dst.get_pointer());
        });
      });

    q.submit([&] (handler &cgh) {
      auto acc = d_dst.template get_access<sycl_read>(cgh);
      cgh.copy(acc, h_dst);
    }).wait();

    T *min = std::min_element( h_dst, h_dst + w*h );
    T *max = std::max_element( h_dst, h_dst + w*h );
    printf("value range [%e, %e]\n", *min, *max);
  }

  free(h_dst);
  free(h_src);
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  int w = atoi(argv[2]);
  int h = atoi(argv[3]);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  surfelRenderTest<float>(q, n, w, h);
  surfelRenderTest<double>(q, n, w, h);
  return 0;
}
