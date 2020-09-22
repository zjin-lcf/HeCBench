#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>

#define restrict __restrict

#define max(a,b) ((a<b)?b:a)
#define min(a,b) ((a<b)?a:b)

const int WSIZE = 12000;           // Walker
const int NSIZE = 2003;           // Values
const int MSIZE = NSIZE*3+3;      // Gradient vectors
const int OSIZE = NSIZE*9+9;      // Hessian Matrices 

const int NSIZE_round = NSIZE%16 ? NSIZE+16-NSIZE%16: NSIZE;
const size_t SSIZE = (size_t)NSIZE_round*48*48*48;  //Coefs size 


static inline void eval_UBspline_3d_s_vgh (
    const float * restrict coefs_init,
    const intptr_t xs,
    const intptr_t ys,
    const intptr_t zs,
    float * restrict vals,
    float * restrict grads,
    float * restrict hess,
    const float *   a, const float *   b, const float *   c,
    const float *  da, const float *  db, const float *  dc,
    const float * d2a, const float * d2b, const float * d2c,
    const float dxInv, const float dyInv, const float dzInv);

void  eval_abc(const float * restrict Af, float tx, float * restrict a);

void bspline (
    const float *restrict spline_coefs,
    const intptr_t xs,
    const intptr_t ys,
    const intptr_t zs,
    float *restrict walkers_vals, 
    float *restrict walkers_grads,
    float *restrict walkers_hess,
    const float* a,
    const float* b,
    const float* c,
    const float* da,
    const float* db,
    const float* dc,
    const float* d2a,
    const float* d2b,
    const float* d2c,
    const float spline_x_grid_delta_inv, 
    const float spline_y_grid_delta_inv, 
    const float spline_z_grid_delta_inv,
    const int spline_num_splines,
    const int i,
    const int ix,
    const int iy,
    const int iz
,
    sycl::nd_item<3> item_ct1) 
{
  int n = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
  if (n < spline_num_splines)
    eval_UBspline_3d_s_vgh ( 
        spline_coefs+ix*xs+iy*ys+iz*zs+n,
        xs, ys, zs, 
        walkers_vals+i*NSIZE+n,
        walkers_grads+i*MSIZE+n*3,
        walkers_hess+i*OSIZE+n*9,
        a,
        b,
        c,
        da,
        db,
        dc,
        d2a,
        d2b,
        d2c,
        spline_x_grid_delta_inv,
        spline_y_grid_delta_inv,
        spline_z_grid_delta_inv );
}

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  float *Af = (float*) malloc (sizeof(float)*16);
  float *dAf = (float*) malloc (sizeof(float)*16);
  float *d2Af = (float*) malloc (sizeof(float)*16);

  Af[0]=-0.166667;
  Af[1]=0.500000;
  Af[2]=-0.500000;
  Af[3]=0.166667;
  Af[4]=0.500000;
  Af[5]=-1.000000;
  Af[6]=0.000000;
  Af[7]=0.666667;
  Af[8]=-0.500000;
  Af[9]=0.500000;
  Af[10]=0.500000;
  Af[11]=0.166667;
  Af[12]=0.166667;
  Af[13]=0.000000;
  Af[14]=0.000000;
  Af[15]=0.000000;
  dAf[0]=0.000000; d2Af[0]=0.000000;
  dAf[1]=-0.500000; d2Af[1]=0.000000;
  dAf[2]=1.000000; d2Af[2]=-1.000000;
  dAf[3]=-0.500000; d2Af[3]=1.000000;
  dAf[4]=0.000000; d2Af[4]=0.000000;
  dAf[5]=1.500000; d2Af[5]=0.000000;
  dAf[6]=-2.000000; d2Af[6]=3.000000;
  dAf[7]=0.000000; d2Af[7]=-2.000000;
  dAf[8]=0.000000; d2Af[8]=0.000000;
  dAf[9]=-1.500000; d2Af[9]=0.000000;
  dAf[10]=1.000000; d2Af[10]=-3.00000;
  dAf[11]=0.500000; d2Af[11]=1.000000;
  dAf[12]=0.000000; d2Af[12]=0.000000;
  dAf[13]=0.500000; d2Af[13]=0.000000;
  dAf[14]=0.000000; d2Af[14]=1.000000;
  dAf[15]=0.000000; d2Af[15]=0.000000;


  float x=0.822387;  
  float y=0.989919;  
  float z=0.104573;

  float* walkers_vals = (float*) malloc(sizeof(float)*WSIZE*NSIZE);
  float* walkers_grads = (float*) malloc(sizeof(float)*WSIZE*MSIZE);
  float* walkers_hess = (float*) malloc(sizeof(float)*WSIZE*OSIZE);
  float* walkers_x = (float*) malloc(sizeof(float)*WSIZE);
  float* walkers_y = (float*) malloc(sizeof(float)*WSIZE);
  float* walkers_z = (float*) malloc(sizeof(float)*WSIZE);

  for (int i=0; i<WSIZE; i++) {
    walkers_x[i] = x + i*1.0/WSIZE;
    walkers_y[i] = y + i*1.0/WSIZE;
    walkers_z[i] = z + i*1.0/WSIZE;
  }

  float* spline_coefs = (float*) malloc (sizeof(float)*SSIZE);
  for(size_t i=0;i<SSIZE;i++)
    spline_coefs[i] = sqrt(0.22 + i * 1.0) * sin(i * 1.0);

  int spline_num_splines = NSIZE;
  int spline_x_grid_start = 0; 
  int spline_y_grid_start = 0; 
  int spline_z_grid_start = 0; 
  int spline_x_grid_num = 45; 
  int spline_y_grid_num = 45; 
  int spline_z_grid_num = 45; 
  int spline_x_stride=NSIZE_round*48*48;
  int spline_y_stride=NSIZE_round*48;
  int spline_z_stride=NSIZE_round;
  int spline_x_grid_delta_inv=45;
  int spline_y_grid_delta_inv=45;
  int spline_z_grid_delta_inv=45;



  float* d_walkers_vals;
  d_walkers_vals =
      (float *)sycl::malloc_device(sizeof(float) * WSIZE * NSIZE, q_ct1);
  q_ct1.memcpy(d_walkers_vals, walkers_vals, sizeof(float) * WSIZE * NSIZE);

  float* d_walkers_grads;
  d_walkers_grads =
      (float *)sycl::malloc_device(sizeof(float) * WSIZE * MSIZE, q_ct1);
  q_ct1.memcpy(d_walkers_grads, walkers_grads, sizeof(float) * WSIZE * MSIZE);

  float* d_walkers_hess;
  d_walkers_hess =
      (float *)sycl::malloc_device(sizeof(float) * WSIZE * OSIZE, q_ct1);
  q_ct1.memcpy(d_walkers_hess, walkers_hess, sizeof(float) * WSIZE * OSIZE);

  float* d_spline_coefs;
  d_spline_coefs = sycl::malloc_device<float>(SSIZE, q_ct1);
  q_ct1.memcpy(d_spline_coefs, spline_coefs, sizeof(float) * SSIZE);

  float* d_a;
  d_a = sycl::malloc_device<float>(4, q_ct1);
  float* d_b;
  d_b = sycl::malloc_device<float>(4, q_ct1);
  float* d_c;
  d_c = sycl::malloc_device<float>(4, q_ct1);
  float* d_da;
  d_da = sycl::malloc_device<float>(4, q_ct1);
  float* d_db;
  d_db = sycl::malloc_device<float>(4, q_ct1);
  float* d_dc;
  d_dc = sycl::malloc_device<float>(4, q_ct1);
  float* d_d2a;
  d_d2a = sycl::malloc_device<float>(4, q_ct1);
  float* d_d2b;
  d_d2b = sycl::malloc_device<float>(4, q_ct1);
  float* d_d2c;
  d_d2c = sycl::malloc_device<float>(4, q_ct1);

  for(int i=0; i<WSIZE; i++){
    float x = walkers_x[i], y = walkers_y[i], z = walkers_z[i];

    float ux = x*spline_x_grid_delta_inv;
    float uy = y*spline_y_grid_delta_inv;
    float uz = z*spline_z_grid_delta_inv;
    float ipartx, iparty, ipartz, tx, ty, tz;
    float a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
    intptr_t xs = spline_x_stride;
    intptr_t ys = spline_y_stride;
    intptr_t zs = spline_z_stride;

    x -= spline_x_grid_start;
    y -= spline_y_grid_start;
    z -= spline_z_grid_start;
    ipartx = (int) ux; tx = ux-ipartx;    int ix = min(max(0,(int) ipartx),spline_x_grid_num-1);
    iparty = (int) uy; ty = uy-iparty;    int iy = min(max(0,(int) iparty),spline_y_grid_num-1);
    ipartz = (int) uz; tz = uz-ipartz;    int iz = min(max(0,(int) ipartz),spline_z_grid_num-1);

    eval_abc(Af,tx,&a[0]);
    q_ct1.memcpy(d_a, a, sizeof(float) * 4);

    eval_abc(Af,ty,&b[0]);
    q_ct1.memcpy(d_b, b, sizeof(float) * 4);

    eval_abc(Af,tz,&c[0]);
    q_ct1.memcpy(d_c, c, sizeof(float) * 4);

    eval_abc(dAf,tx,&da[0]);
    q_ct1.memcpy(d_da, da, sizeof(float) * 4);

    eval_abc(dAf,ty,&db[0]);
    q_ct1.memcpy(d_db, db, sizeof(float) * 4);

    eval_abc(dAf,tz,&dc[0]);
    q_ct1.memcpy(d_dc, dc, sizeof(float) * 4);

    eval_abc(d2Af,tx,&d2a[0]);
    q_ct1.memcpy(d_d2a, d2a, sizeof(float) * 4);

    eval_abc(d2Af,ty,&d2b[0]);
    q_ct1.memcpy(d_d2b, d2b, sizeof(float) * 4);

    eval_abc(d2Af,tz,&d2c[0]);
    q_ct1.memcpy(d_d2c, d2c, sizeof(float) * 4);

    sycl::range<3> global_size((spline_num_splines + 255) / 256 * 256, 1, 1);
    sycl::range<3> local_size(256, 1, 1);

    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = global_size * local_size;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(local_size.get(2), local_size.get(1),
                                           local_size.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            bspline(d_spline_coefs, xs, ys, zs, d_walkers_vals, d_walkers_grads,
                    d_walkers_hess, d_a, d_b, d_c, d_da, d_db, d_dc, d_d2a,
                    d_d2b, d_d2c, spline_x_grid_delta_inv,
                    spline_y_grid_delta_inv, spline_z_grid_delta_inv,
                    spline_num_splines, i, ix, iy, iz, item_ct1);
          });
    });
    dev_ct1.queues_wait_and_throw();
  }

  q_ct1.memcpy(walkers_vals, d_walkers_vals, sizeof(float) * WSIZE * NSIZE);
  q_ct1.memcpy(walkers_grads, d_walkers_grads, sizeof(float) * WSIZE * MSIZE);
  q_ct1.memcpy(walkers_hess, d_walkers_hess, sizeof(float) * WSIZE * OSIZE);
  dev_ct1.queues_wait_and_throw();

  // collect results for the first walker
  float resVal = 0.0;
  float resGrad = 0.0;
  float resHess = 0.0;

  for( int i = 0; i < NSIZE; i++ ) resVal = resVal + walkers_vals[i];
  for( int i = 0; i < MSIZE; i++ ) resGrad = resGrad + walkers_grads[i];
  for( int i = 0; i < OSIZE; i++ ) resHess = resHess + walkers_hess[i];
  printf("walkers[0]->collect([resVal resGrad resHess]) = [%e %e %e]\n",
      resVal,resGrad, resHess);  

  free(Af);
  free(dAf);
  free(d2Af);
  free(walkers_vals);
  free(walkers_grads);
  free(walkers_hess);
  free(walkers_x);
  free(walkers_y);
  free(walkers_z);
  free(spline_coefs);

  return 0;
}




void  eval_abc(const float * restrict Af, float tx, float * restrict a){

  a[0] = ( ( Af[0]  * tx + Af[1] ) * tx + Af[2] ) * tx + Af[3];
  a[1] = ( ( Af[4]  * tx + Af[5] ) * tx + Af[6] ) * tx + Af[7];
  a[2] = ( ( Af[8]  * tx + Af[9] ) * tx + Af[10] ) * tx + Af[11];
  a[3] = ( ( Af[12] * tx + Af[13] ) * tx + Af[14] ) * tx + Af[15];
}


static inline void eval_UBspline_3d_s_vgh (
    const float * restrict coefs_init,
    const intptr_t xs,
    const intptr_t ys,
    const intptr_t zs,
    float * restrict vals,
    float * restrict grads,
    float * restrict hess,
    const float *   a, const float *   b, const float *   c,
    const float *  da, const float *  db, const float *  dc,
    const float * d2a, const float * d2b, const float * d2c,
    const float dxInv, const float dyInv, const float dzInv)
{
  float h[9];
  float v0 = 0.0f;
  for (int i = 0; i < 9; ++i) h[i] = 0.0f;

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++){
      float pre20 = d2a[i]*  b[j];
      float pre10 =  da[i]*  b[j];
      float pre00 =   a[i]*  b[j];
      float pre11 =  da[i]* db[j];
      float pre01 =   a[i]* db[j];
      float pre02 =   a[i]*d2b[j];

      const float * restrict coefs = coefs_init + i*xs + j*ys;

      float sum0 =   c[0] * coefs[0] +   c[1] * coefs[zs] +   c[2] * coefs[zs*2] +   c[3] * coefs[zs*3];
      float sum1 =  dc[0] * coefs[0] +  dc[1] * coefs[zs] +  dc[2] * coefs[zs*2] +  dc[3] * coefs[zs*3];
      float sum2 = d2c[0] * coefs[0] + d2c[1] * coefs[zs] + d2c[2] * coefs[zs*2] + d2c[3] * coefs[zs*3];

      h[0]  += pre20 * sum0;
      h[1]  += pre11 * sum0;
      h[2]  += pre10 * sum1;
      h[4]  += pre02 * sum0;
      h[5]  += pre01 * sum1;
      h[8]  += pre00 * sum2;
      h[3]  += pre10 * sum0;
      h[6]  += pre01 * sum0;
      h[7]  += pre00 * sum1;
      v0    += pre00 * sum0;
    }
  vals[0] = v0;
  grads[0]  = h[3] * dxInv;
  grads[1]  = h[6] * dyInv;
  grads[2]  = h[7] * dzInv;

  hess [0] = h[0]*dxInv*dxInv;
  hess [1] = h[1]*dxInv*dyInv;
  hess [2] = h[2]*dxInv*dzInv;
  hess [3] = h[1]*dxInv*dyInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [4] = h[4]*dyInv*dyInv;
  hess [5] = h[5]*dyInv*dzInv;
  hess [6] = h[2]*dxInv*dzInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [7] = h[5]*dyInv*dzInv; //Copy hessian elements into lower half of 3x3 matrix
  hess [8] = h[8]*dzInv*dzInv;
}
