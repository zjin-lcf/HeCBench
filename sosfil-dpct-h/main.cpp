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

#define DPCT_USM_LEVEL_NONE
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
  dpct::dpct_malloc((void **)&d_sos, sizeof(T) * sos_size);
  dpct::dpct_memcpy(d_sos, sos, sizeof(T) * sos_size, dpct::host_to_device);

  // initial  conditions
  const int z_size = (n_sections + 1) * blocks * zi_width;
  T* zi = (T*) malloc (sizeof(T) * z_size);
  for (int i = 0; i < z_size; i++) zi[i] = (T)1; // for test

  T* d_zi;
  dpct::dpct_malloc((void **)&d_zi, sizeof(T) * z_size);
  dpct::dpct_memcpy(d_zi, zi, sizeof(T) * z_size, dpct::host_to_device);

  // input signals
  const int x_size = n_signals * n_samples;
  T* x = (T*) malloc (sizeof(T) * x_size);
  for (int i = 0; i < n_signals; i++) 
    for (int j = 0; j < n_samples; j++)
      x[i * n_samples + j] = (T)std::sin(2 * 3.14 * (i + 1 + j));

  T* d_x;
  dpct::dpct_malloc((void **)&d_x, sizeof(T) * x_size);
  dpct::dpct_memcpy(d_x, x, sizeof(T) * x_size, dpct::host_to_device);

  const int out_size = n_sections;
  const int shared_mem = (out_size + z_size + sos_size) * sizeof(T); 

  for (int n = 0; n < 100; n++)
  {
    std::pair<dpct::buffer_t, size_t> d_sos_buf_ct4 =
        dpct::get_buffer_and_offset(d_sos);
    size_t d_sos_offset_ct4 = d_sos_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_zi_buf_ct5 =
        dpct::get_buffer_and_offset(d_zi);
    size_t d_zi_offset_ct5 = d_zi_buf_ct5.second;
    std::pair<dpct::buffer_t, size_t> d_x_buf_ct6 =
        dpct::get_buffer_and_offset(d_x);
    size_t d_x_offset_ct6 = d_x_buf_ct6.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          dpct_local_acc_ct1(sycl::range<1>(shared_mem), cgh);
      auto d_sos_acc_ct4 =
          d_sos_buf_ct4.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_zi_acc_ct5 =
          d_zi_buf_ct5.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_x_acc_ct6 =
          d_x_buf_ct6.first.get_access<sycl::access::mode::read_write>(cgh);

      auto dpct_global_range = blocksPerGrid * threadsPerBlock;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threadsPerBlock.get(2), threadsPerBlock.get(1),
                             threadsPerBlock.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            T *d_sos_ct4 = (T *)(&d_sos_acc_ct4[0] + d_sos_offset_ct4);
            T *d_zi_ct5 = (T *)(&d_zi_acc_ct5[0] + d_zi_offset_ct5);
            T *d_x_ct6 = (T *)(&d_x_acc_ct6[0] + d_x_offset_ct6);
            sosfilt<T>(n_signals, n_samples, n_sections, zi_width, d_sos_ct4,
                       d_zi_ct5, d_x_ct6, item_ct1,
                       dpct_local_acc_ct1.get_pointer());
          });
    });
  }

  dpct::dpct_memcpy(x, d_x, sizeof(T) * x_size, dpct::device_to_host);
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
  dpct::dpct_free(d_x);
  dpct::dpct_free(d_sos);
  dpct::dpct_free(d_zi);
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
