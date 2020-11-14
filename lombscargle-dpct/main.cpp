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

///////////////////////////////////////////////////////////////////////////////
//                            LOMBSCARGLE                                    //
///////////////////////////////////////////////////////////////////////////////

/*
   import cupy as cp
   import matplotlib.pyplot as plt
   First define some input parameters for the signal:
   A = 2.
   w = 1.
   phi = 0.5 * cp.pi
   nin = 10000
   nout = 1000000
   r = cp.random.rand(nin)
   x = cp.linspace(0.01, 10*cp.pi, nin)
   Plot a sine wave for the selected times:
   y = A * cp.sin(w*x+phi)
   Define the array of frequencies for which to compute the periodogram:
   f = cp.linspace(0.01, 10, nout)
   Calculate Lomb-Scargle periodogram:
   pgram = cusignal.lombscargle(x, y, f, normalize=True)
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

void       
lombscargle( const int x_shape,
    const int freqs_shape,
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ freqs,
    float *__restrict__ pgram,
    const float y_dot ,
    sycl::nd_item<3> item_ct1) {

  const int tx = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2));
  const int stride =
      (item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2));

  for ( int tid = tx; tid < freqs_shape; tid += stride ) {

    float freq = freqs[tid] ;

    float xc = 0;
    float xs = 0;
    float cc = 0;
    float ss = 0;
    float cs = 0;
    float c;
    float s; 

    for ( int j = 0; j < x_shape; j++ ) {
      /*
      DPCT1017:1: The sycl::sincos call is used instead of the sincosf call.
      These two calls do not provide exactly the same functionality. Check the
      potential precision and/or performance issues for the generated code.
      */
      s = sycl::sincos(
          freq * x[j],
          sycl::make_ptr<float, sycl::access::address_space::global_space>(&c));
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }

    float c_tau;
    float s_tau;
    float tau = sycl::atan2(2.0f * cs, cc - ss) / (2.0f * freq);
    /*
    DPCT1017:0: The sycl::sincos call is used instead of the sincosf call. These
    two calls do not provide exactly the same functionality. Check the potential
    precision and/or performance issues for the generated code.
    */
    s_tau = sycl::sincos(
        freq * tau,
        sycl::make_ptr<float, sycl::access::address_space::global_space>(
            &c_tau));
    float c_tau2 = c_tau * c_tau ;
    float s_tau2 = s_tau * s_tau ;
    float cs_tau = 2.0f * c_tau * s_tau ;

    pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) * ( c_tau * xc + s_tau * xs ) /
            ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
          ( ( c_tau * xs - s_tau * xc ) * ( c_tau * xs - s_tau * xc ) /
            ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) * y_dot;
  }
}

void lombscargle_cpu( const int x_shape,
    const int freqs_shape,
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ freqs,
    float *__restrict__ pgram,
    const float y_dot ) {


  for ( int tid = 0; tid < freqs_shape; tid ++) {

    float freq = freqs[tid] ;
    float xc = 0;
    float xs = 0;
    float cc = 0;
    float ss = 0;
    float cs = 0;
    float c;
    float s; 

    for ( int j = 0; j < x_shape; j++ ) {
      sincosf(freq * x[j], &s, &c);
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }

    float c_tau;
    float s_tau;
    float tau = atan2f(2.0f * cs, cc - ss) / (2.0f * freq);
    sincosf(freq * tau, &s_tau, &c_tau);
    float c_tau2 = c_tau * c_tau ;
    float s_tau2 = s_tau * s_tau ;
    float cs_tau = 2.0f * c_tau * s_tau ;

    pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) * ( c_tau * xc + s_tau * xs ) /
            ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
          ( ( c_tau * xs - s_tau * xc ) * ( c_tau * xs - s_tau * xc ) /
            ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) * y_dot;
  }
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  const int x_shape = 1000;
  const int freqs_shape = 100000;
  const float A = 2.f;
  const float w = 1.0f;
  const float phi = 1.57f; 

  float* x = (float*) malloc (sizeof(float)*x_shape); 
  float* y = (float*) malloc (sizeof(float)*x_shape); 
  float* f = (float*) malloc (sizeof(float)*freqs_shape); 
  float* p  = (float*) malloc (sizeof(float)*freqs_shape); 
  float* p2 = (float*) malloc (sizeof(float)*freqs_shape); 

  for (int i = 0; i < x_shape; i++)
    x[i] = 0.01f + i*(31.4f - 0.01f)/x_shape;

  for (int i = 0; i < x_shape; i++)
    y[i] = A * sinf(w * x[i] + phi);

  for (int i = 0; i < freqs_shape; i++)
    f[i] = 0.01f + i*(10.f-0.01f)/freqs_shape;

  const float y_dot = 2.0f/1.5f;
  float* d_x; 
  float* d_y; 
  float* d_f; 
  float* d_p;
  d_x = sycl::malloc_device<float>(x_shape, q_ct1);
  d_y = sycl::malloc_device<float>(x_shape, q_ct1);
  d_f = sycl::malloc_device<float>(freqs_shape, q_ct1);
  d_p = sycl::malloc_device<float>(freqs_shape, q_ct1);
  q_ct1.memcpy(d_x, x, sizeof(float) * x_shape).wait();
  q_ct1.memcpy(d_y, y, sizeof(float) * x_shape).wait();
  q_ct1.memcpy(d_f, f, sizeof(float) * freqs_shape).wait();

  sycl::range<3> grids((freqs_shape + 255) / 256 * 256, 1, 1);
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
            lombscargle(x_shape, freqs_shape, d_x, d_y, d_f, d_p, y_dot,
                        item_ct1);
          });
    });

  q_ct1.memcpy(p, d_p, sizeof(float) * freqs_shape).wait();

  // verification
  lombscargle_cpu(x_shape, freqs_shape, x, y, f, p2, y_dot);

  bool error = false;
  for (int i = 0; i < freqs_shape; i++) {
    if (fabsf(p[i] - p2[i]) > 1e-3f) {
      printf("%.3f %.3f\n", p[i], p2[i]);
      error = true;
      break;
    }
  }
  if (error) 
    printf("Fail\n");
  else
    printf("Pass\n");

  sycl::free(d_x, q_ct1);
  sycl::free(d_y, q_ct1);
  sycl::free(d_f, q_ct1);
  sycl::free(d_p, q_ct1);
  free(x);
  free(y);
  free(f);
  free(p);
  free(p2);
  return 0;
}

