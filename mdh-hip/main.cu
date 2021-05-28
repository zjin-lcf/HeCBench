/*
 * A set of simple Multiple Debye-Huckel (MDH) kernels 
 * inspired by APBS:
 *   http://www.poissonboltzmann.org/ 
 *
 * This code was all originally written by David Gohara on MacOS X, 
 * and has been subsequently been modified by John Stone, porting to Linux,
 * adding vectorization, and several other platform-specific 
 * performance optimizations.
 * 
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "WKFUtils.h"

#define SEP printf("-----------------------------------------------------------\n")

inline __host__ __device__ float4 sqrtf(float4 v)
{
  return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}

inline __host__ __device__ float4 expf(float4 v)
{
  return make_float4(expf(v.x), expf(v.y), expf(v.z), expf(v.w));
}

void gendata(float *ax,float *ay,float *az,
    float *gx,float *gy,float *gz,
    float *charge,float *size,int natom,int ngrid) {

  int i;    

  printf("Generating Data.. \n");
  for (i=0; i<natom; i++) {
    ax[i] = ((float) rand() / (float) RAND_MAX);
    ay[i] = ((float) rand() / (float) RAND_MAX);
    az[i] = ((float) rand() / (float) RAND_MAX);
    charge[i] = ((float) rand() / (float) RAND_MAX);
    size[i] = ((float) rand() / (float) RAND_MAX);
  }

  for (i=0; i<ngrid; i++) {
    gx[i] = ((float) rand() / (float) RAND_MAX);
    gy[i] = ((float) rand() / (float) RAND_MAX);
    gz[i] = ((float) rand() / (float) RAND_MAX);
  }
  printf("Done generating inputs.\n\n");
}

void print_total(float * arr, int ngrid){
  int i;
  double accum = 0.0;
  for (i=0; i<ngrid; i++){
    accum += arr[i];
  }
  printf("Accumulated value: %1.7g\n",accum);
}

__global__ void mdh (
    const int ngrid,
    const int natom,
    const int ngadj,
    const float *ax, 
    const float *ay,
    const float *az,
    const float *gx, 
    const float *gy, 
    const float *gz,
    const float *charge, 
    const float *size, 
    const float xkappa, 
    const float pre1, 
    float *val)
{
  HIP_DYNAMIC_SHARED(float, shared);

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid * 5    ] = ax[jatom + lid];
      shared[lid * 5 + 1] = ay[jatom + lid];
      shared[lid * 5 + 2] = az[jatom + lid];
      shared[lid * 5 + 3] = charge[jatom + lid];
      shared[lid * 5 + 4] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i * 5    ];
      float4 dy = lgy - shared[i * 5 + 1];
      float4 dz = lgz - shared[i * 5 + 2];
      float4 dist = sqrtf(dx*dx + dy*dy + dz*dz);
      v += pre1 * (shared[i * 5 + 3] / dist)  *
        expf( -xkappa * (dist - shared[i * 5 + 4])) /
        (1.0f + xkappa * shared[i * 5 + 4]);
    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}

__global__ void mdh2 (
    const int ngrid,
    const int natom,
    const int ngadj,
    const float *ax, 
    const float *ay,
    const float *az,
    const float *gx, 
    const float *gy, 
    const float *gz,
    const float *charge, 
    const float *size, 
    const float xkappa, 
    const float pre1, 
    float *val)
{
  HIP_DYNAMIC_SHARED(float, shared);

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid          ] = ax[jatom + lid];
      shared[lid +   lsize] = ay[jatom + lid];
      shared[lid + 2*lsize] = az[jatom + lid];
      shared[lid + 3*lsize] = charge[jatom + lid];
      shared[lid + 4*lsize] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i          ];
      float4 dy = lgy - shared[i +   lsize];
      float4 dz = lgz - shared[i + 2*lsize];
      float4 dist = sqrtf( dx * dx + dy * dy + dz * dz );
      v += pre1 * ( shared[i + 3*lsize] / dist )  *
        expf( -xkappa * (dist - shared[i + 4*lsize])) /
        (1.0f + xkappa * shared[i + 4*lsize]);

    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}

void run_gpu_kernel(
    const bool smem_strided_write,
    const int wgsize, 
    const int itmax,
    const int ngrid,
    const int natom,
    const int ngadj,
    const float *ax, 
    const float *ay,
    const float *az,
    const float *gx, 
    const float *gy, 
    const float *gz,
    const float *charge, 
    const float *size, 
    const float xkappa, 
    const float pre1, 
    float *val)
{
  wkf_timerhandle timer = wkf_timer_create();

  //Allocate memory for programs and kernels
  float* d_ax;
  hipMalloc((void**)&d_ax, sizeof(float)*natom);
  hipMemcpy(d_ax, ax, sizeof(float)*natom, hipMemcpyHostToDevice);

  float* d_ay;
  hipMalloc((void**)&d_ay, sizeof(float)*natom);
  hipMemcpy(d_ay, ay, sizeof(float)*natom, hipMemcpyHostToDevice);

  float* d_az;
  hipMalloc((void**)&d_az, sizeof(float)*natom);
  hipMemcpy(d_az, az, sizeof(float)*natom, hipMemcpyHostToDevice);

  float* d_charge;
  hipMalloc((void**)&d_charge, sizeof(float)*natom);
  hipMemcpy(d_charge, charge, sizeof(float)*natom, hipMemcpyHostToDevice);

  float* d_size;
  hipMalloc((void**)&d_size, sizeof(float)*natom);
  hipMemcpy(d_size, size, sizeof(float)*natom, hipMemcpyHostToDevice);

  float* d_gx;
  hipMalloc((void**)&d_gx, sizeof(float)*ngadj);
  hipMemcpy(d_gx, gx, sizeof(float)*ngadj, hipMemcpyHostToDevice);

  float* d_gy;
  hipMalloc((void**)&d_gy, sizeof(float)*ngadj);
  hipMemcpy(d_gy, gy, sizeof(float)*ngadj, hipMemcpyHostToDevice);

  float* d_gz;
  hipMalloc((void**)&d_gz, sizeof(float)*ngadj);
  hipMemcpy(d_gz, gz, sizeof(float)*ngadj, hipMemcpyHostToDevice);

  float* d_val;
  hipMalloc((void**)&d_val, sizeof(float)*ngadj);

  wkf_timer_start(timer); 

  // set work-item dimensions 
  // scale number of work units by vector size
  dim3 grids (ngadj / 4 / wgsize);
  dim3 blocks (wgsize);

  for(int n = 0; n < itmax; n++) {
    if (smem_strided_write) 
      hipLaunchKernelGGL(mdh, dim3(grids), dim3(blocks), sizeof(float)*5*wgsize, 0, 
          ngrid,
          natom,
          ngadj,
          d_ax,
          d_ay,
          d_az,
          d_gx,
          d_gy,
          d_gz,
          d_charge,
          d_size,
          xkappa,
          pre1,
          d_val
          );
    else 
      hipLaunchKernelGGL(mdh2, dim3(grids), dim3(blocks), sizeof(float)*5*wgsize, 0, 
          ngrid,
          natom,
          ngadj,
          d_ax,
          d_ay,
          d_az,
          d_gx,
          d_gy,
          d_gz,
          d_charge,
          d_size,
          xkappa,
          pre1,
          d_val
          );
  }
  hipDeviceSynchronize();

  wkf_timer_stop(timer);
  double avg_kernel_time = wkf_timer_time(timer) / ((double) itmax);
  printf("Average kernel time on the device: %1.12g\n", avg_kernel_time);

  // read output image
  hipMemcpy(val, d_val, sizeof(float)*ngrid, hipMemcpyDeviceToHost);

  hipFree(d_ax);
  hipFree(d_ay);
  hipFree(d_az);
  hipFree(d_gx);
  hipFree(d_gy);
  hipFree(d_gz);
  hipFree(d_val);
  hipFree(d_size);
  hipFree(d_charge);

  wkf_timer_destroy(timer);
}


// reference CPU kernel
void run_cpu_kernel(
    const int ngrid,
    const int natom,
    const float *ax,
    const float *ay,
    const float *az,
    const float *gx,
    const float *gy,
    const float *gz,
    const float *charge,
    const float *size,
    const float xkappa,
    const float pre1,
    float *val)
{

  #pragma omp parallel for
  for(int igrid=0;igrid<ngrid;igrid++){
    float sum = 0.0f;
    #pragma omp parallel for simd reduction(+:sum)
    for(int iatom=0; iatom<natom; iatom++) {
      float dist = sqrtf((gx[igrid]-ax[iatom])*(gx[igrid]-ax[iatom]) + 
          (gy[igrid]-ay[iatom])*(gy[igrid]-ay[iatom]) + 
          (gz[igrid]-az[iatom])*(gz[igrid]-az[iatom]));

      sum += pre1*(charge[iatom]/dist)*expf(-xkappa*(dist-size[iatom]))
        / (1+xkappa*size[iatom]);
    }
    val[igrid] = sum;
  }
}


void usage() {
  printf("command line parameters:\n");
  printf("Optional test flags:\n");
  printf("  -itmax N         loop test N times\n");
  printf("  -wgsize          set workgroup size\n");
}

void getargs(int argc, const char **argv, int *itmax, int *wgsize) {
  int i;
  for (i=0; i<argc; i++) {
    if ((!strcmp(argv[i], "-itmax")) && ((i+1) < argc)) {
      i++;
      *itmax = atoi(argv[i]);
    }

    if ((!strcmp(argv[i], "-wgsize")) && ((i+1) < argc)) {
      i++;
      *wgsize = atoi(argv[i]);
    }
  }

  printf("Run parameters:\n");
  printf("  kernel loop count: %d\n", *itmax);
  printf("     workgroup size: %d\n", *wgsize);
}


int main(int argc, const char **argv) {
  int itmax = 100;
  int wgsize = 256;

  getargs(argc, argv, &itmax, &wgsize);

  wkf_timerhandle timer = wkf_timer_create();

  int natom = 5877;
  int ngrid = 134918;
  int ngadj = ngrid + (512 - (ngrid & 511));

  float pre1 = 4.46184985145e19;
  float xkappa = 0.0735516324639;

  float *ax = (float*)calloc(natom, sizeof(float));
  float *ay = (float*)calloc(natom, sizeof(float));
  float *az = (float*)calloc(natom, sizeof(float));
  float *charge = (float*)calloc(natom, sizeof(float));
  float *size = (float*)calloc(natom, sizeof(float));

  float *gx = (float*)calloc(ngadj, sizeof(float));
  float *gy = (float*)calloc(ngadj, sizeof(float));
  float *gz = (float*)calloc(ngadj, sizeof(float));

  // host result
  float *val1 = (float*)calloc(ngadj, sizeof(float));
  // device result
  float *val2 = (float*)calloc(ngadj, sizeof(float));

  gendata(ax, ay, az, gx, gy, gz, charge, size, natom, ngrid);

  wkf_timer_start(timer);
  run_cpu_kernel(ngadj, natom, ax, ay, az, gx, gy, gz, charge, size, xkappa, pre1, val1);
  wkf_timer_stop(timer);

  SEP;
  print_total(val1, ngrid);
  printf("CPU Time: %1.12g\n", wkf_timer_time(timer));
  SEP;

  wkf_timer_start(timer);
  run_gpu_kernel(true, wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz, 
      charge, size, xkappa, pre1, val2);
  wkf_timer_stop(timer);

  SEP;
  print_total(val2, ngrid);
  printf("GPU Time: %1.12g\n", wkf_timer_time(timer)); // Vec4 Optimized: 
  SEP;

  wkf_timer_start(timer);
  run_gpu_kernel(false, wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz, 
      charge, size, xkappa, pre1, val2);
  wkf_timer_stop(timer);

  SEP;
  print_total(val2, ngrid);
  printf("GPU Time: %1.12g\n", wkf_timer_time(timer)); // Vec4 Optimized: 
  SEP;

  free(ax);
  free(ay);
  free(az);
  free(charge);
  free(size);
  free(gx);
  free(gy);
  free(gz);
  free(val1);
  free(val2);

  wkf_timer_destroy(timer);

  return 0;
}
