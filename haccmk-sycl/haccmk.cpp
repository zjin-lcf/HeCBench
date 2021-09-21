#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "common.h"

template <typename T> 
class HACCmk;

template <typename T>
void haccmk (
    const size_t n,// global size
    const int ilp, // inner loop count
    const T fsrrmax,
    const T mp_rsm,
    const T fcoeff,
    const T*__restrict xx, 
    const T*__restrict yy,
    const T*__restrict zz,
    const T*__restrict mass,
          T*__restrict vx2,
          T*__restrict vy2,
          T*__restrict vz2 ) 
{
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  range<1> numOfItems{n};
  const  property_list props = {property::buffer::use_host_ptr()};
  buffer<T, 1> buf_xx(xx, ilp, props);
  buffer<T, 1> buf_yy(yy, ilp, props);
  buffer<T, 1> buf_zz(zz, ilp, props);
  buffer<T, 1> buf_mass(mass, ilp, props);
  buffer<T, 1> buf_vx2(vx2, numOfItems, props);
  buffer<T, 1> buf_vy2(vy2, numOfItems, props);
  buffer<T, 1> buf_vz2(vz2, numOfItems, props);

  q.submit([&](handler& cgh) {
    auto acc_xx     = buf_xx.template get_access<sycl_read>(cgh);
    auto acc_yy     = buf_yy.template get_access<sycl_read>(cgh);
    auto acc_zz     = buf_zz.template get_access<sycl_read>(cgh);
    auto acc_mass   = buf_mass.template get_access<sycl_read>(cgh);
    auto acc_vx2    = buf_vx2.template get_access<sycl_read_write>(cgh);
    auto acc_vy2    = buf_vy2.template get_access<sycl_read_write>(cgh);
    auto acc_vz2    = buf_vz2.template get_access<sycl_read_write>(cgh);

    cgh.parallel_for<class HACCmk<T>>(numOfItems, [=](id<1> i) {
      const float ma0 = 0.269327f; 
      const float ma1 = -0.0750978f; 
      const float ma2 = 0.0114808f; 
      const float ma3 = -0.00109313f; 
      const float ma4 = 0.0000605491f; 
      const float ma5 = -0.00000147177f;

      float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

      xi = 0.f; 
      yi = 0.f;
      zi = 0.f;

      float xxi = acc_xx[i];
      float yyi = acc_yy[i];
      float zzi = acc_zz[i];

      for ( int j = 0; j < ilp; j++ ) {
        dxc = acc_xx[j] - xxi;
        dyc = acc_yy[j] - yyi;
        dzc = acc_zz[j] - zzi;

        r2 = dxc * dxc + dyc * dyc + dzc * dzc;

        if ( r2 < fsrrmax ) m = acc_mass[j]; else m = 0.f;

        f = r2 + mp_rsm;
        f = m * ( 1.f / ( f * cl::sycl::sqrt( f ) ) - 
            ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

        xi = xi + f * dxc;
        yi = yi + f * dyc;
        zi = zi + f * dzc;
      }

      acc_vx2[i] = acc_vx2[i] + xi * fcoeff;
      acc_vy2[i] = acc_vy2[i] + yi * fcoeff;
      acc_vz2[i] = acc_vz2[i] + zi * fcoeff;
    });
  });
}

void haccmk_gold(
    int count1,
    float xxi,
    float yyi,
    float zzi,
    float fsrrmax2,
    float mp_rsm2, 
    float *__restrict xx1, 
    float *__restrict yy1, 
    float *__restrict zz1, 
    float *__restrict mass1, 
    float *__restrict dxi,
    float *__restrict dyi,
    float *__restrict dzi )
{
  const float ma0 = 0.269327f, 
              ma1 = -0.0750978f, 
              ma2 = 0.0114808f,
              ma3 = -0.00109313f,
              ma4 = 0.0000605491f,
              ma5 = -0.00000147177f;

  float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

  xi = 0.f; 
  yi = 0.f;
  zi = 0.f;

  for (int j = 0; j < count1; j++ ) {
    dxc = xx1[j] - xxi;
    dyc = yy1[j] - yyi;
    dzc = zz1[j] - zzi;

    r2 = dxc * dxc + dyc * dyc + dzc * dzc;

    if ( r2 < fsrrmax2 ) m = mass1[j]; else m = 0.f;

    f = r2 + mp_rsm2;
    f =  m * ( 1.f / ( f * sqrtf( f ) ) - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

    xi = xi + f * dxc;
    yi = yi + f * dyc;
    zi = zi + f * dzc;
  }

  *dxi = xi;
  *dyi = yi;
  *dzi = zi;
}


int main( int argc, char *argv[] )
{
  float fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1, dx2, dy2, dz2;
  int n1, n2, i;
  n1 = 784;
  n2 = 15000;
  printf( "Outer loop count is set %d\n", n1 );
  printf( "Inner loop count is set %d\n", n2 );

  float* xx = (float*) malloc (sizeof(float) * n2);
  float* yy = (float*) malloc (sizeof(float) * n2);
  float* zz = (float*) malloc (sizeof(float) * n2);
  float* mass = (float*) malloc (sizeof(float) * n2);
  float* vx2 = (float*) malloc (sizeof(float) * n2);
  float* vy2 = (float*) malloc (sizeof(float) * n2);
  float* vz2 = (float*) malloc (sizeof(float) * n2);
  float* vx2_hw = (float*) malloc (sizeof(float) * n2);
  float* vy2_hw = (float*) malloc (sizeof(float) * n2);
  float* vz2_hw = (float*) malloc (sizeof(float) * n2);

  /* Initial data preparation */
  fcoeff = 0.23f;  
  fsrrmax2 = 0.5f; 
  mp_rsm2 = 0.03f;
  dx1 = 1.0f/(float)n2;
  dy1 = 2.0f/(float)n2;
  dz1 = 3.0f/(float)n2;
  xx[0] = 0.f;
  yy[0] = 0.f;
  zz[0] = 0.f;
  mass[0] = 2.f;

  for ( i = 1; i < n2; i++ ) {
    xx[i] = xx[i-1] + dx1;
    yy[i] = yy[i-1] + dy1;
    zz[i] = zz[i-1] + dz1;
    mass[i] = (float)i * 0.01f + xx[i];
  }

  for ( i = 0; i < n2; i++ ) {
    vx2[i] = 0.f;
    vy2[i] = 0.f;
    vz2[i] = 0.f;
    vx2_hw[i] = 0.f; 
    vy2_hw[i] = 0.f; 
    vz2_hw[i] = 0.f;
  }

  for ( i = 0; i < n1; ++i) {
    haccmk_gold( n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx2, &dy2, &dz2 );    
    vx2[i] = vx2[i] + dx2 * fcoeff;
    vy2[i] = vy2[i] + dy2 * fcoeff;
    vz2[i] = vz2[i] + dz2 * fcoeff;
  }

  haccmk(n1, n2, fsrrmax2, mp_rsm2, fcoeff, xx,
      yy, zz, mass, vx2_hw, vy2_hw, vz2_hw); 

  // verify
  int error = 0;
  const float eps = 1e-1f;
  for (i = 0; i < n2; i++) {
    if (fabsf(vx2[i] - vx2_hw[i]) > eps) {
      printf("error at vx2[%d] %f %f\n", i, vx2[i], vx2_hw[i]);
      error = 1;
      break;
    }
    if (fabsf(vy2[i] - vy2_hw[i]) > eps) {
      printf("error at vy2[%d]: %f %f\n", i, vy2[i], vy2_hw[i]);
      error = 1;
      break;
    }
    if (fabsf(vz2[i] - vz2_hw[i]) > eps) {
      printf("error at vz2[%d]: %f %f\n", i, vz2[i], vz2_hw[i]);
      error = 1;
      break;
    }
  } 

  free(xx);
  free(yy);
  free(zz);
  free(mass);
  free(vx2);
  free(vy2);
  free(vz2);
  free(vx2_hw);
  free(vy2_hw);
  free(vz2_hw);
  if (error) {
    printf("FAIL\n"); 
    return EXIT_FAILURE; 
  } else {
    printf("PASS\n"); 
    return EXIT_SUCCESS;
  }
}


