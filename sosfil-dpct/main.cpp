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

///////////////////////////////////////////////////////////////////////////////
//                                SOSFILT                                    //
///////////////////////////////////////////////////////////////////////////////
#define MAX_THREADS 256
#define THREADS 32  
#define sos_width  6   // https://www.mathworks.com/help/signal/ref/sosfilt.html 

template<typename T>
void sosfilt( 
    const int n_signals,
    const int n_samples,
    const int n_sections,
    const int zi_width,
    const T *__restrict__ sos,
    const T *__restrict__ zi,
    T *__restrict__ x_in,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
  auto smem = (char *)dpct_local;
  T *s_out = reinterpret_cast<T *>( smem );
  T *s_zi = reinterpret_cast<T *>( &s_out[n_sections] ) ;
  T *s_sos = reinterpret_cast<T *>( &s_zi[n_sections * zi_width] ) ;

  // dim3 blocksPerGrid (1, blocks);
  // dim3 threadsPerBlock (256, 1);
  const int tx = static_cast<int>(item_ct1.get_local_id(2));
  const int ty = static_cast<int>(item_ct1.get_group(1) *
                                      item_ct1.get_local_range().get(1) +
                                  item_ct1.get_local_id(1));

  // Reset shared memory
  s_out[tx] = 0;

  // Load zi
  for ( int i = 0; i < zi_width; i++ ) {
    s_zi[tx * zi_width + i] = zi[ty * n_sections * zi_width + tx * zi_width + i];
  }

  // Load SOS
#pragma unroll 
  for ( int i = 0; i < sos_width; i++ ) {
    s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
  }

  item_ct1.barrier();

  const int load_size = n_sections - 1 ;
  const int unload_size = n_samples - load_size ;

  T temp;
  T x_n;

  if ( ty < n_signals ) {
    // Loading phase
    for ( int n = 0; n < load_size; n++ ) {
      if ( tx == 0 ) {
        x_n = x_in[ty * n_samples + n];
      } else {
        x_n = s_out[tx - 1];
      }

      // Use direct II transposed structure
      temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

      s_zi[tx * zi_width + 0] =
        s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

      s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

      s_out[tx] = temp;

      item_ct1.barrier();
    }

    // Processing phase
    for ( int n = load_size; n < n_samples; n++ ) {
      if ( tx == 0 ) {
        x_n = x_in[ty * n_samples + n];
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
        x_in[ty * n_samples + ( n - load_size )] = temp;
      }

      item_ct1.barrier();
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
          x_in[ty * n_samples + ( n + unload_size )] = temp;
        }
        item_ct1.barrier();
      }
    }
  }
}

  template <typename T>
void filtering (const int n_signals, const int n_samples, const int n_sections, const int zi_width)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  // the number of second-order sections must be less than max threads per block
  assert(MAX_THREADS >= n_sections);

  // The number of samples must be greater than or equal to the number of sections
  assert(n_samples >= n_sections);


  // randomize input data
  srand(2);

  const int blocks = n_signals;

  sycl::range<3> blocksPerGrid(1, blocks, 1);
  sycl::range<3> threadsPerBlock(THREADS, 1, 1);

  // Second-order section digital filter
  const int sos_size = n_sections * sos_width ;

  T* sos = (T*) malloc (sizeof(T) * sos_size);
  for (int i = 0; i < n_sections; i++)
    for (int j = 0; j < sos_width; j++)
      sos[i*sos_width+j] = (T)1 ; // for test 

  T* d_sos;
  d_sos = (T *)sycl::malloc_device(sizeof(T) * sos_size, q_ct1);
  q_ct1.memcpy(d_sos, sos, sizeof(T) * sos_size).wait();

  // initial  conditions
  const int z_size = (n_sections + 1) * blocks * zi_width;
  T* zi = (T*) malloc (sizeof(T) * z_size);
  for (int i = 0; i < z_size; i++) zi[i] = (T)1; // for test

  T* d_zi;
  d_zi = (T *)sycl::malloc_device(sizeof(T) * z_size, q_ct1);
  q_ct1.memcpy(d_zi, zi, sizeof(T) * z_size).wait();

  // input signals
  const int x_size = n_signals * n_samples;
  T* x = (T*) malloc (sizeof(T) * x_size);
  for (int i = 0; i < n_signals; i++) 
    for (int j = 0; j < n_samples; j++)
      x[i * n_samples + j] = (T)std::sin(2 * 3.14 * (i + 1 + j));

  T* d_x;
  d_x = (T *)sycl::malloc_device(sizeof(T) * x_size, q_ct1);
  q_ct1.memcpy(d_x, x, sizeof(T) * x_size).wait();

  const int out_size = n_sections;
  const int shared_mem = (out_size + z_size + sos_size) * sizeof(T); 

  for (int n = 0; n < 100; n++)
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          dpct_local_acc_ct1(sycl::range<1>(shared_mem), cgh);

      auto dpct_global_range = blocksPerGrid * threadsPerBlock;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threadsPerBlock.get(2), threadsPerBlock.get(1),
                             threadsPerBlock.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            sosfilt<T>(n_signals, n_samples, n_sections, zi_width, d_sos, d_zi,
                       d_x, item_ct1, dpct_local_acc_ct1.get_pointer());
          });
    });

  q_ct1.memcpy(x, d_x, sizeof(T) * x_size).wait();
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
  sycl::free(d_x, q_ct1);
  sycl::free(d_sos, q_ct1);
  sycl::free(d_zi, q_ct1);
}

int main(int argc, char** argv) {

  const int numSections = THREADS; 

#ifdef DEBUG
  const int numSignals = 2; 
  const int numSamples = THREADS+1;
#else
  // failed to launch the double-precision kernel when numSignals = 16 on a P100 GPU
  const int numSignals = 8;  
  const int numSamples = 100000;
#endif

  const int zi_width = 2;
  filtering<float> (numSignals, numSamples, numSections, zi_width);
  filtering<double> (numSignals, numSamples, numSections, zi_width);
  return 0;
}
