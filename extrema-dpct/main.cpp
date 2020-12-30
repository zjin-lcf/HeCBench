// Copyright (c) 2019-2020, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

__dpct_inline__ void clip_plus(const bool &clip, const int &n, int &plus) {
  if ( clip ) {
    if ( plus >= n ) {
      plus = n - 1;
    }
  } else {
    if ( plus >= n ) {
      plus -= n;
    }
  }
}

__dpct_inline__ void clip_minus(const bool &clip, const int &n, int &minus) {
  if ( clip ) {
    if ( minus < 0 ) {
      minus = 0;
    }
  } else {
    if ( minus < 0 ) {
      minus += n;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 1D                                //
///////////////////////////////////////////////////////////////////////////////

  template<typename T>
void relextrema_1D( const int  n,
    const int  order,
    const bool clip,
    const T *__restrict__ inp,
    bool *__restrict__ results,
    sycl::nd_item<3> item_ct1)
{

  const int tx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
  const int stride =
      item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);

  for ( int tid = tx; tid < n; tid += stride ) {

    const T data = inp[tid];
    bool    temp = true;

    for ( int o = 1; o < ( order + 1 ); o++ ) {
      int plus = tid + o;
      int minus = tid - o;

      clip_plus( clip, n, plus );
      clip_minus( clip, n, minus );

      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp;
  }
}

  template<typename T>
void cpu_relextrema_1D( const int  n,
    const int  order,
    const bool clip,
    const T *__restrict__ inp,
    bool *__restrict__ results)
{

  for ( int tid = 0; tid < n; tid++ ) {

    const T data = inp[tid];
    bool    temp = true;

    for ( int o = 1; o < ( order + 1 ); o++ ) {
      int plus = tid + o;
      int minus = tid - o;

      clip_plus( clip, n, plus );
      clip_minus( clip, n, minus );

      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp;
  }
}


  template<typename T>
void relextrema_2D( const int  in_x,
    const int  in_y,
    const int  order,
    const bool clip,
    const int  axis,
    const T *__restrict__ inp,
    bool *__restrict__ results,
    sycl::nd_item<3> item_ct1) 
{

  const int ty = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
  const int tx = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                 item_ct1.get_local_id(1);

  if ( ( tx < in_y ) && ( ty < in_x ) ) {
    int tid = tx * in_x + ty ;

    const T data = inp[tid] ;
    bool    temp = true ;

    for ( int o = 1; o < ( order + 1 ); o++ ) {

      int plus;
      int minus;

      if ( axis == 0 ) {
        plus  = tx + o;
        minus = tx - o;

        clip_plus( clip, in_y, plus );
        clip_minus( clip, in_y, minus );

        plus  = plus * in_x + ty;
        minus = minus * in_x + ty;
      } else {
        plus  = ty + o;
        minus = ty - o;

        clip_plus( clip, in_x, plus );
        clip_minus( clip, in_x, minus );

        plus  = tx * in_x + plus;
        minus = tx * in_x + minus;
      }

      temp &= data > inp[plus] ;
      temp &= data >= inp[minus] ;
    }
    results[tid] = temp;
  }
}

template<typename T>
void cpu_relextrema_2D( const int  in_x,
    const int  in_y,
    const int  order,
    const bool clip,
    const int  axis,
    const T *__restrict__ inp,
    bool *__restrict__ results) 
{
  for (int tx = 0; tx < in_y; tx++)
    for (int ty = 0; ty < in_x; ty++) {

      int tid = tx * in_x + ty ;

      const T data = inp[tid] ;
      bool    temp = true ;

      for ( int o = 1; o < ( order + 1 ); o++ ) {

        int plus;
        int minus;

        if ( axis == 0 ) {
          plus  = tx + o;
          minus = tx - o;

          clip_plus( clip, in_y, plus );
          clip_minus( clip, in_y, minus );

          plus  = plus * in_x + ty;
          minus = minus * in_x + ty;
        } else {
          plus  = ty + o;
          minus = ty - o;

          clip_plus( clip, in_x, plus );
          clip_minus( clip, in_x, minus );

          plus  = tx * in_x + plus;
          minus = tx * in_x + minus;
        }

        temp &= data > inp[plus] ;
        temp &= data >= inp[minus] ;
      }
      results[tid] = temp;
    }
}

template <typename T>
void test_1D (const int length, const int order, const bool clip)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  T* x = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;
  
  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);

  T* d_x;
  bool *d_result;
  d_x = (T *)sycl::malloc_device(length * sizeof(T), q_ct1);
  q_ct1.memcpy(d_x, x, length * sizeof(T)).wait();
  d_result = sycl::malloc_device<bool>(length, q_ct1);

  sycl::range<3> grids((length + 255) / 256, 1, 1);
  sycl::range<3> threads(256, 1, 1);

  for (int n = 0; n < 100; n++)
    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            relextrema_1D<T>(length, order, clip, d_x, d_result, item_ct1);
          });
    });

  q_ct1.memcpy(gpu_r, d_result, length * sizeof(bool)).wait();

  cpu_relextrema_1D<T>(length, order, clip, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++)
    if (cpu_r[i] != gpu_r[i]) {
      error = 1; 
      break;
    }

  sycl::free(d_x, q_ct1);
  sycl::free(d_result, q_ct1);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("1D test: FAILED\n");
}

template <typename T>
void test_2D (const int length_x, const int length_y, const int order,
              const bool clip, const int axis)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  const int length = length_x * length_y;
  T* x = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;
  
  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);

  T* d_x;
  bool *d_result;
  d_x = (T *)sycl::malloc_device(length * sizeof(T), q_ct1);
  q_ct1.memcpy(d_x, x, length * sizeof(T)).wait();
  d_result = sycl::malloc_device<bool>(length, q_ct1);

  sycl::range<3> grids((length_x + 15) / 16, (length_y + 15) / 16, 1);
  sycl::range<3> threads(16, 16, 1);

  for (int n = 0; n < 100; n++)
    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            relextrema_2D(length_x, length_y, order, clip, axis, d_x, d_result,
                          item_ct1);
          });
    });

  q_ct1.memcpy(gpu_r, d_result, length * sizeof(bool)).wait();

  cpu_relextrema_2D(length_x, length_y, order, clip, axis, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++)
    if (cpu_r[i] != gpu_r[i]) {
      error = 1; 
      break;
    }

  sycl::free(d_x, q_ct1);
  sycl::free(d_result, q_ct1);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("2D test: FAILED\n");
}

int main() {

  for (int order = 1; order <= 128; order = order * 2) {
    test_1D<int>(1000000, order, true);
    test_1D<long>(1000000, order, true);
    test_1D<float>(1000000, order, true);
    test_1D<double>(1000000, order, true);
  }

  for (int order = 1; order <= 128; order = order * 2) {
    test_2D<int>(1000, 1000, order, true, 1);
    test_2D<long>(1000, 1000, order, true, 1);
    test_2D<float>(1000, 1000, order, true, 1);
    test_2D<double>(1000, 1000, order, true, 1);
  }

  for (int order = 1; order <= 128; order = order * 2) {
    test_2D<int>(1000, 1000, order, true, 0);
    test_2D<long>(1000, 1000, order, true, 0);
    test_2D<float>(1000, 1000, order, true, 0);
    test_2D<double>(1000, 1000, order, true, 0);
  }

  return 0;
}

