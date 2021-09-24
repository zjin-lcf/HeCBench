#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

#include "reference.cpp"

__global__ void smoothingFilter(
    int Lx, int Ly, 
    int Threshold, int MaxRad, 
    const float*__restrict Img,
            int*__restrict Box,
          float*__restrict Norm)
{
  int tid = threadIdx.x;
  int tjd = threadIdx.y;
  int i = blockIdx.x * blockDim.x + tid;
  int j = blockIdx.y * blockDim.y + tjd;
  int stid = tjd * blockDim.x + tid;
  int gtid = j * Lx + i;  

  // part of shared memory may be unused
  __shared__ float s_Img[1024];

  if ( i < Lx && j < Ly )
    s_Img[stid] = Img[gtid];

  __syncthreads();

  if ( i < Lx && j < Ly )
  {
    // Smoothing parameters
    float sum = 0.f;
    int q = 1;
    int s = q;
    int ksum = 0;

    // Continue until parameters are met
    while (sum < Threshold && q < MaxRad)
    {
      s = q;
      sum = 0.f;
      ksum = 0;

      // Normal adaptive smoothing
      for (int ii = -s; ii < s+1; ii++)
        for (int jj = -s; jj < s+1; jj++)
          if ( (i-s >= 0) && (i+s < Ly) && (j-s >= 0) && (j+s < Lx) )
          {
            ksum++;
            // Compute within bounds of block dimensions
            if( tid-s >= 0 && tid+s < blockDim.x && tjd-s >= 0 && tjd+s < blockDim.y )
              sum += s_Img[stid + ii*blockDim.x + jj];
            // Compute block borders with global memory
            else
              sum += Img[gtid + ii*Lx + jj];
          }
      q++;
    }
    Box[gtid] = s;

    // Normalization for each box
    for (int ii = -s; ii < s+1; ii++)
      for (int jj = -s; jj < s+1; jj++)
        if (ksum != 0) 
          atomicAdd(&Norm[gtid + ii*Lx + jj], __fdividef(1.f, (float)ksum));
  }
}

__global__ void normalizeFilter(int Lx, int Ly, float*__restrict Img, const float*__restrict Norm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ( i < Lx && j < Ly ) {
    int gtid = j * Lx + i;  
    const float norm = Norm[gtid];
    if (norm != 0) Img[gtid] = __fdividef(Img[gtid], norm);
  }
}

__global__ void outFilter( 
    int Lx, int Ly,
    const float*__restrict Img,
    const   int*__restrict Box,
          float*__restrict Out )
{
  int tid = threadIdx.x;
  int tjd = threadIdx.y;
  int i = blockIdx.x * blockDim.x + tid;
  int j = blockIdx.y * blockDim.y + tjd;
  int stid = tjd * blockDim.x + tid;
  int gtid = j * Lx + i;  

  // part of shared memory may be unused
  __shared__ float s_Img[1024];

  if ( i < Lx && j < Ly )
    s_Img[stid] = Img[gtid];

  __syncthreads();

  if ( i < Lx && j < Ly )
  {
    const int s = Box[gtid];
    float sum = 0.f;
    int ksum  = 0;

    for (int ii = -s; ii < s+1; ii++)
      for (int jj = -s; jj < s+1; jj++)
        if ( (i-s >= 0) && (i+s < Lx) && (j-s >= 0) && (j+s < Ly) )
        {
          ksum++;
          if( tid-s >= 0 && tid+s < blockDim.x && tjd-s >= 0 && tjd+s < blockDim.y )
            sum += s_Img[stid + ii*blockDim.y + jj];
          else
            sum += Img[gtid + ii*Ly + jj];
        }
    if ( ksum != 0 ) Out[gtid] = __fdividef(sum , (float)ksum);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
     printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
     exit(1);
  }

  // only a square image is supported
  const int Lx = atoi(argv[1]);
  const int Ly = Lx;
  const int size = Lx * Ly;

  const int Threshold = atoi(argv[2]);
  const int MaxRad = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
 
  // input image
  float *img = (float*) malloc (sizeof(float) * size);

  // host and device results
  float *norm = (float*) malloc (sizeof(float) * size);
  float *h_norm = (float*) malloc (sizeof(float) * size);

  int *box = (int*) malloc (sizeof(int) * size);
  int *h_box = (int*) malloc (sizeof(int) * size);

  float *out = (float*) malloc (sizeof(float) * size);
  float *h_out = (float*) malloc (sizeof(float) * size);

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = rand() % 256;
    norm[i] = box[i] = out[i] = 0;
  }

  float *d_img;
  cudaMalloc((void**)&d_img, sizeof(float) * size);

  float *d_norm;
  cudaMalloc((void**)&d_norm, sizeof(float) * size);

  int *d_box;
  cudaMalloc((void**)&d_box, sizeof(int) * size);

  float *d_out;
  cudaMalloc((void**)&d_out, sizeof(float) * size);

  dim3 grids ((Lx+15)/16, (Ly+15)/16);
  dim3 blocks (16, 16);

  // reset output
  cudaMemcpy(d_out, out, sizeof(float) * size, cudaMemcpyHostToDevice);

  for (int i = 0; i < repeat; i++) {
    // restore input image
    cudaMemcpy(d_img, img, sizeof(float) * size, cudaMemcpyHostToDevice);
    // reset norm
    cudaMemcpy(d_norm, norm, sizeof(float) * size, cudaMemcpyHostToDevice);
    // launch three kernels
    smoothingFilter<<<grids, blocks>>>(Lx, Ly, Threshold, MaxRad, d_img, d_box, d_norm);
    normalizeFilter<<<grids, blocks>>>(Lx, Ly, d_img, d_norm);
    outFilter<<<grids, blocks>>>(Lx, Ly, d_img, d_box, d_out);
  }

  cudaMemcpy(out, d_out, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(box, d_box, sizeof(int) * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(norm, d_norm, sizeof(float) * size, cudaMemcpyDeviceToHost);

  // verify
  reference (Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  cudaFree(d_img);
  cudaFree(d_norm);
  cudaFree(d_box);
  cudaFree(d_out);
  free(img);
  free(norm);
  free(h_norm);
  free(box);
  free(h_box);
  free(out);
  free(h_out);
  return 0;
}
