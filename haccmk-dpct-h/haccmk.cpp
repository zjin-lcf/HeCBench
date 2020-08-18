#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void
haccmk_kernel ( const int n1,  // outer loop count
		const int n2,  // inner loop count
		const float *xx, 
		const float *yy,
		const float *zz,
		const float *mass,
		float *vx2,
		float *vy2,
		float *vz2,
		const float fsrmax,
		const float mp_rsm,
		const float fcoeff ,
		sycl::nd_item<3> item_ct1) 
{
 int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
         item_ct1.get_local_id(2);
        if (i >= n1) return;

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

	float xxi = xx[i];
	float yyi = yy[i];
	float zzi = zz[i];

	for ( int j = 0; j < n2; j++ ) {
		dxc = xx[j] - xxi;
		dyc = yy[j] - yyi;
		dzc = zz[j] - zzi;

		r2 = dxc * dxc + dyc * dyc + dzc * dzc;

		if ( r2 < fsrmax ) m = mass[j]; else m = 0.f;

		f = r2 + mp_rsm;
  f = m *
      (1.f / (f * sycl::sqrt(f)) -
       (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));

                xi = xi + f * dxc;
		yi = yi + f * dyc;
		zi = zi + f * dzc;
	}

	vx2[i] = vx2[i] + xi * fcoeff;
	vy2[i] = vy2[i] + yi * fcoeff;
	vz2[i] = vz2[i] + zi * fcoeff;
}


void haccmk ( const int n1,
              const int n2,
              const float* xx, 
              const float* yy,
              const float* zz,
              const float* mass,
              float* vx2,
              float* vy2,
              float* vz2,
              const float fsrmax,
              const float mp_rsm,
              const float fcoeff ) {

	int block_size = 256;

	float *d_xx, *d_yy, *d_zz, *d_mass;
	float *d_vx2, *d_vy2, *d_vz2;

 dpct::dpct_malloc((void **)&d_xx, sizeof(float) * n2);
 dpct::dpct_malloc((void **)&d_yy, sizeof(float) * n2);
 dpct::dpct_malloc((void **)&d_zz, sizeof(float) * n2);
 dpct::dpct_malloc((void **)&d_mass, sizeof(float) * n2);
 dpct::dpct_malloc((void **)&d_vx2, sizeof(float) * n1);
 dpct::dpct_malloc((void **)&d_vy2, sizeof(float) * n1);
 dpct::dpct_malloc((void **)&d_vz2, sizeof(float) * n1);

 dpct::dpct_memcpy(d_xx, xx, sizeof(float) * n2, dpct::host_to_device);
 dpct::dpct_memcpy(d_yy, yy, sizeof(float) * n2, dpct::host_to_device);
 dpct::dpct_memcpy(d_zz, zz, sizeof(float) * n2, dpct::host_to_device);
 dpct::dpct_memcpy(d_mass, mass, sizeof(float) * n2, dpct::host_to_device);
 dpct::dpct_memcpy(d_vx2, vx2, sizeof(float) * n1, dpct::host_to_device);
 dpct::dpct_memcpy(d_vy2, vy2, sizeof(float) * n1, dpct::host_to_device);
 dpct::dpct_memcpy(d_vz2, vz2, sizeof(float) * n1, dpct::host_to_device);

 {
  dpct::buffer_t d_xx_buf_ct2 = dpct::get_buffer(d_xx);
  dpct::buffer_t d_yy_buf_ct3 = dpct::get_buffer(d_yy);
  dpct::buffer_t d_zz_buf_ct4 = dpct::get_buffer(d_zz);
  dpct::buffer_t d_mass_buf_ct5 = dpct::get_buffer(d_mass);
  dpct::buffer_t d_vx2_buf_ct6 = dpct::get_buffer(d_vx2);
  dpct::buffer_t d_vy2_buf_ct7 = dpct::get_buffer(d_vy2);
  dpct::buffer_t d_vz2_buf_ct8 = dpct::get_buffer(d_vz2);
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
   auto d_xx_acc_ct2 =
       d_xx_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
   auto d_yy_acc_ct3 =
       d_yy_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
   auto d_zz_acc_ct4 =
       d_zz_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
   auto d_mass_acc_ct5 =
       d_mass_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
   auto d_vx2_acc_ct6 =
       d_vx2_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
   auto d_vy2_acc_ct7 =
       d_vy2_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);
   auto d_vz2_acc_ct8 =
       d_vz2_buf_ct8.get_access<sycl::access::mode::read_write>(cgh);

   cgh.parallel_for(
       sycl::nd_range<3>(
           sycl::range<3>(1, 1, (n1 + block_size - 1) / block_size) *
               sycl::range<3>(1, 1, block_size),
           sycl::range<3>(1, 1, block_size)),
       [=](sycl::nd_item<3> item_ct1) {
        haccmk_kernel(
            n1, n2, (const float *)(&d_xx_acc_ct2[0]),
            (const float *)(&d_yy_acc_ct3[0]),
            (const float *)(&d_zz_acc_ct4[0]),
            (const float *)(&d_mass_acc_ct5[0]), (float *)(&d_vx2_acc_ct6[0]),
            (float *)(&d_vy2_acc_ct7[0]), (float *)(&d_vz2_acc_ct8[0]), fsrmax,
            mp_rsm, fcoeff, item_ct1);
       });
  });
 }

 dpct::dpct_memcpy(vx2, d_vx2, sizeof(float) * n1, dpct::device_to_host);
 dpct::dpct_memcpy(vy2, d_vy2, sizeof(float) * n1, dpct::device_to_host);
 dpct::dpct_memcpy(vz2, d_vz2, sizeof(float) * n1, dpct::device_to_host);
 dpct::dpct_free(d_xx);
 dpct::dpct_free(d_yy);
 dpct::dpct_free(d_zz);
 dpct::dpct_free(d_mass);
 dpct::dpct_free(d_vx2);
 dpct::dpct_free(d_vy2);
 dpct::dpct_free(d_vz2);
}

void haccmk_gold( int n2, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, 
		float *xx1, float *yy1, float *zz1, float *mass1, float *dxi, float *dyi, float *dzi )
{

	int j;
	const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, 
	      ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;
	float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

	xi = 0.f; 
	yi = 0.f;
	zi = 0.f;

	for ( j = 0; j < n2; j++ ) {
		dxc = xx1[j] - xxi;
		dyc = yy1[j] - yyi;
		dzc = zz1[j] - zzi;

		r2 = dxc * dxc + dyc * dyc + dzc * dzc;

		if ( r2 < fsrrmax2 ) m = mass1[j]; else m = 0.f;

		f = r2 + mp_rsm2;
  f = m *
      (1.f / (f * sqrtf(f)) -
       (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));

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

	for ( i = 0; i < n2; i++ ) { // i starts from zero
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
	} // for i


	haccmk(n1, n2, xx, yy, zz, mass, vx2_hw, vy2_hw, vz2_hw, 
			fsrrmax2, mp_rsm2, fcoeff);

	// verifiy
	int error = 0;
	const float eps = 1e-1f;  // 
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
		printf("FAILED\n"); 
		return EXIT_FAILURE;
	} else {
		printf("PASSED\n"); 
		return EXIT_SUCCESS;
	}
}


