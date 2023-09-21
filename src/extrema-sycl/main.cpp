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

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

// Forward declarations
template<typename T>
class extrema1D;

template<typename T>
class extrema2D;

inline void clip_plus( const bool &clip, const int &n, int &plus ) {
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

inline void clip_minus( const bool &clip, const int &n, int &minus ) {
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
void cpu_relextrema_1D( const int  n,
    const int  order,
    const bool clip,
    const T * inp,
    bool * results)
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
void cpu_relextrema_2D( const int  in_x,
    const int  in_y,
    const int  order,
    const bool clip,
    const int  axis,
    const T * inp,
    bool * results) 
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
long test_1D (sycl::queue &q, const int length, const int order, const bool clip,
              const int repeat, const char* type)
{
  T* x = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;

  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);

  T *d_x = sycl::malloc_device<T>(length, q);
  q.memcpy(d_x, x, length * sizeof(T));

  bool *d_result = sycl::malloc_device<bool>(length, q);

  sycl::range<1> gws ((length+255)/256*256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class extrema1D<T>>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        const int tid = item.get_global_id(0);
        if (tid < length) {
          const T data = d_x[tid];
          bool    temp = true;

          #pragma unroll 8
          for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus = tid + o;
            int minus = tid - o;

            clip_plus( clip, length, plus );
            clip_minus( clip, length, minus );

            temp &= data > d_x[plus];
            temp &= data >= d_x[minus];
          }
          d_result[tid] = temp;
        }
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average 1D kernel (type = %s, order = %d, clip = %d) execution time %f (s)\n", 
         type, order, clip, (time * 1e-9f) / repeat);

  q.memcpy(gpu_r, d_result, length * sizeof(bool)).wait();

  cpu_relextrema_1D<T>(length, order, clip, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++)
    if (cpu_r[i] != gpu_r[i]) {
      error = 1; 
      break;
    }

  sycl::free(d_x, q);
  sycl::free(d_result, q);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("1D test: FAILED\n");
  return time;
}

// length_x is the number of columns
// length_y is the number of rows
template <typename T>
long test_2D (sycl::queue &q, const int length_x, const int length_y, 
              const int order, const bool clip, const int axis,
              const int repeat, const char* type) 
{
  const int length = length_x * length_y;
  T* x = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;

  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);

  T *d_x = sycl::malloc_device<T>(length, q);
  q.memcpy(d_x, x, length * sizeof(T));

  bool *d_result = sycl::malloc_device<bool>(length, q);

  sycl::range<2> gws ((length_y+15)/16*16, (length_x+15)/16*16);
  sycl::range<2> lws (16, 16);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class extrema2D<T>>(
      sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        const int ty = item.get_global_id(1); 
        const int tx = item.get_global_id(0);

        if ( ( tx < length_y ) && ( ty < length_x ) ) {
          int tid = tx * length_x + ty ;
          const T data = d_x[tid] ;
          bool    temp = true ;
          #pragma unroll 8
          for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus;
            int minus;
            if ( axis == 0 ) {
              plus  = tx + o;
              minus = tx - o;

              clip_plus( clip, length_y, plus );
              clip_minus( clip, length_y, minus );

              plus  = plus * length_x + ty;
              minus = minus * length_x + ty;
            } else {
              plus  = ty + o;
              minus = ty - o;

              clip_plus( clip, length_x, plus );
              clip_minus( clip, length_x, minus );

              plus  = tx * length_x + plus;
              minus = tx * length_x + minus;
            }
            temp &= data > d_x[plus] ;
            temp &= data >= d_x[minus] ;
          }
          d_result[tid] = temp;
        }
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average 2D kernel (type = %s, order = %d, clip = %d, axis = %d) execution time %f (s)\n", 
       type, order, clip, axis, (time * 1e-9f) / repeat);
  
  q.memcpy(gpu_r, d_result, length * sizeof(bool)).wait();

  cpu_relextrema_2D(length_x, length_y, order, clip, axis, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++)
    if (cpu_r[i] != gpu_r[i]) {
      error = 1; 
      break;
    }

  sycl::free(d_x, q);
  sycl::free(d_result, q);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("2D test: FAILED\n");
  return time;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage ./%s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  long time = 0;

  for (int order = 1; order <= 128; order = order * 2) {
    time += test_1D<   int>(q, 1000000, order, true, repeat, "int");
    time += test_1D<  long>(q, 1000000, order, true, repeat, "long");
    time += test_1D< float>(q, 1000000, order, true, repeat, "float");
    time += test_1D<double>(q, 1000000, order, true, repeat, "double");
  }

  for (int order = 1; order <= 128; order = order * 2) {
    time += test_2D<   int>(q, 1000, 1000, order, true, 1, repeat, "int");
    time += test_2D<  long>(q, 1000, 1000, order, true, 1, repeat, "long");
    time += test_2D< float>(q, 1000, 1000, order, true, 1, repeat, "float");
    time += test_2D<double>(q, 1000, 1000, order, true, 1, repeat, "double");
  }

  for (int order = 1; order <= 128; order = order * 2) {
    time += test_2D<   int>(q, 1000, 1000, order, true, 0, repeat, "int");
    time += test_2D<  long>(q, 1000, 1000, order, true, 0, repeat, "long");
    time += test_2D< float>(q, 1000, 1000, order, true, 0, repeat, "float");
    time += test_2D<double>(q, 1000, 1000, order, true, 0, repeat, "double");
  }

  printf("\n-----------------------------------------------\n");
  printf("Total kernel execution time: %lf (s)", time * 1e-9);
  printf("\n-----------------------------------------------\n");

  return 0;
}
