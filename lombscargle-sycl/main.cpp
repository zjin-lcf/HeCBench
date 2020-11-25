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

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "common.h"


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
      sincosf( freq * x[j], &s, &c );
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }

    float c_tau;
    float s_tau;
    float tau = atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) ;
    sincosf( freq * tau, &s_tau, &c_tau );
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
    y[i] = A * sinf(w*x[i]+phi);

  for (int i = 0; i < freqs_shape; i++)
    f[i] = 0.01f + i*(10.f-0.01f)/freqs_shape;

  const float y_dot = 2.0f/1.5f;

  {
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    buffer<float,1> d_x (x, x_shape);
    buffer<float,1> d_y (y, x_shape);
    buffer<float,1> d_f (f, freqs_shape);
    buffer<float,1> d_p (p, freqs_shape);

    const float y_dot = 2.0f/1.5f;

    range<1> gws ((freqs_shape+255)/256*256);
    range<1> lws (256);

    for (int n = 0; n < 100; n++) {
      q.submit([&] (handler &cgh) {
        auto p = d_p.template get_access<sycl_discard_write>(cgh);
        auto x = d_x.template get_access<sycl_read>(cgh);
        auto y = d_y.template get_access<sycl_read>(cgh);
        auto f = d_f.template get_access<sycl_read>(cgh);
        cgh.parallel_for<class lombscargle>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          const int tx = item.get_global_id(0);
	  const int stride = item.get_local_range(0) * item.get_group_range(0);
          for ( int tid = tx; tid < freqs_shape; tid += stride ) {
            float freq = f[tid] ;
            float xc = 0;
            float xs = 0;
            float cc = 0;
            float ss = 0;
            float cs = 0;
            float c;
            float s; 

            for ( int j = 0; j < x_shape; j++ ) {
              s = cl::sycl::sincos( freq * x[j],  &c );
              xc += y[j] * c;
              xs += y[j] * s;
              cc += c * c;
              ss += s * s;
              cs += c * s;
            }

            float c_tau;
            float s_tau;
            float tau = cl::sycl::atan2( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) ;
            s_tau = cl::sycl::sincos( freq * tau, &c_tau );
            float c_tau2 = c_tau * c_tau ;
            float s_tau2 = s_tau * s_tau ;
            float cs_tau = 2.0f * c_tau * s_tau ;

            p[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) * ( c_tau * xc + s_tau * xs ) /
                    ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
                  ( ( c_tau * xs - s_tau * xc ) * ( c_tau * xs - s_tau * xc ) /
                    ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) * y_dot;
          }
        });
      });
    }
    q.wait();
  }

  // verification
  lombscargle_cpu(x_shape, freqs_shape, x, y, f, p2, y_dot);

  bool error = false;
  for (int i = 0; i < freqs_shape; i++) {
    if (fabsf(p[i]-p2[i]) > 1e-3f) {
      printf("%.3f %.3f\n", p[i], p2[i]);
      error = true;
      break;
    }
  }
  if (error) 
    printf("Fail\n");
  else
    printf("Pass\n");

  free(x);
  free(y);
  free(f);
  free(p);
  free(p2);
  return 0;
}

