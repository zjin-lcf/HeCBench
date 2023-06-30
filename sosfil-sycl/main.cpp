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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////////////
//                                SOSFILT                                    //
///////////////////////////////////////////////////////////////////////////////
#define MAX_THREADS 256
#define THREADS 32
#define sos_width  6   // https://www.mathworks.com/help/signal/ref/sosfilt.html

// Forward declarations
template <typename T>
class sosfilter;

template <typename T>
void filtering (sycl::queue &q, const int repeat,
                const int n_signals, const int n_samples,
                const int n_sections, const int zi_width)
{
  // the number of second-order sections must be less than max threads per block
  assert(MAX_THREADS >= n_sections);

  // The number of samples must be greater than or equal to the number of sections
  assert(n_samples >= n_sections);

  // randomize input data
  srand(2);

  const int blocks = n_signals;

  // Second-order section digital filter
  const int sos_size = n_sections * sos_width ;

  T* sos = (T*) malloc (sizeof(T) * sos_size);
  for (int i = 0; i < n_sections; i++)
    for (int j = 0; j < sos_width; j++)
      sos[i*sos_width+j] = (T)1 ; // for test

  // initial  conditions
  const int z_size = (n_sections + 1) * blocks * zi_width;
  T* zi = (T*) malloc (sizeof(T) * z_size);
  for (int i = 0; i < z_size; i++) zi[i] = (T)1; // for test

  // input signals
  const int x_size = n_signals * n_samples;
  T* x = (T*) malloc (sizeof(T) * x_size);
  for (int i = 0; i < n_signals; i++)
    for (int j = 0; j < n_samples; j++)
      x[i*n_samples+j] = (T)std::sin(2*3.14*(i+1+j));

  T *d_sos = sycl::malloc_device<T>(sos_size, q);
  q.memcpy(d_sos, sos, sizeof(T) * sos_size);

  T *d_zi = sycl::malloc_device<T>(z_size, q);
  q.memcpy(d_zi, zi, sizeof(T) * z_size);

  T *d_x = sycl::malloc_device<T>(x_size, q);
  q.memcpy(d_x, x, sizeof(T) * x_size);

  sycl::range<2> gws (blocks, THREADS);
  sycl::range<2> lws (1, THREADS);

  const int out_size = n_sections;
  const int shared_mem_size = (out_size + z_size + sos_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<T, 1> s_out (sycl::range<1>(shared_mem_size), cgh);
      cgh.parallel_for<class sosfilter<T>>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {

        T *s_zi =  &s_out[n_sections] ;
        T *s_sos = &s_zi[n_sections * zi_width] ;

        const int tx = static_cast<int>( item.get_local_id(1) ) ;
        const int ty = static_cast<int>( item.get_global_id(0) ) ;

        // Reset shared memory
        s_out[tx] = 0;

        // Load zi
        for ( int i = 0; i < zi_width; i++ ) {
          s_zi[tx * zi_width + i] = d_zi[ty * n_sections * zi_width + tx * zi_width + i];
        }

        // Load SOS
        #pragma unroll
        for ( int i = 0; i < sos_width; i++ ) {
          s_sos[tx * sos_width + i] = d_sos[tx * sos_width + i];
        }

        item.barrier(sycl::access::fence_space::local_space);

        const int load_size = n_sections - 1 ;
        const int unload_size = n_samples - load_size ;

        T temp;
        T x_n;

        if ( ty < n_signals ) {
          // Loading phase
          for ( int n = 0; n < load_size; n++ ) {
            if ( tx == 0 ) {
              x_n = d_x[ty * n_samples + n];
            } else {
              x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

            s_zi[tx * zi_width + 0] =
              s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

            s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            s_out[tx] = temp;

            item.barrier(sycl::access::fence_space::local_space);
          }

          // Processing phase
          for ( int n = load_size; n < n_samples; n++ ) {
            if ( tx == 0 ) {
              x_n = d_x[ty * n_samples + n];
            } else {
              x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

            s_zi[tx * zi_width + 0] =
              s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

            s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            if ( tx < load_size ) {
              s_out[tx] = temp;
            } else {
              d_x[ty * n_samples + ( n - load_size )] = temp;
            }

            item.barrier(sycl::access::fence_space::local_space);
          }

          // Unloading phase
          for ( int n = 0; n < n_sections; n++ ) {
            // retire threads that are less than n
            if ( tx > n ) {
              x_n = s_out[tx - 1];

              // Use direct II transposed structure
              temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

              s_zi[tx * zi_width + 0] =
                s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

              s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

              if ( tx < load_size ) {
                s_out[tx] = temp;
              } else {
                d_x[ty * n_samples + ( n + unload_size )] = temp;
              }
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
        }
      });
    });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %lf (s)\n", time * 1e-9 / repeat);

  q.memcpy(x, d_x, sizeof(T) * x_size).wait();

#ifdef DEBUG
  for (int i = 0; i < n_signals; i++) {
    for (int j = 0; j < n_samples; j++)
      printf("%.2f ", x[i*n_samples+j]);
    printf("\n");
  }
#endif

  free(x);
  free(sos);
  free(zi);
  sycl::free(d_x, q);
  sycl::free(d_sos, q);
  sycl::free(d_zi, q);
}

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int numSections = THREADS;

#ifdef DEBUG
  const int numSignals = 2;
  const int numSamples = THREADS+1;
#else
  // failed to launch the double-precision kernel when numSignals = 16 on a P100 GPU
  const int numSignals = 8;
  const int numSamples = 100000;
#endif

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int zi_width = 2;
  filtering<float> (q, repeat, numSignals, numSamples, numSections, zi_width);
  filtering<double> (q, repeat, numSignals, numSamples, numSections, zi_width);
  return 0;
}
