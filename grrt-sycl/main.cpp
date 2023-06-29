/***********************************************************************************
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License

  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 

 ***********************************************************************************/

#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <sycl/sycl.hpp>
#include "constants.h"

#include "kernels.cpp"

int main()
{
  // a set of variables defined in constants.h
  double  VariablesIn[VarINNUM];

  A           = 0.;    // black hole spin
  INCLINATION = acos(0.25)/PI*180.;     // inclination angle in unit of degree                    
  SIZE        = IMAGE_SIZE;
  printf("task1: image size = %d  x  %d  pixels\n",IMAGE_SIZE,IMAGE_SIZE);

  // number of grids; the coordinate of each grid is given by (GridIdxX,GridIdY)
  int ImaDimX, ImaDimY; 

  // number of blocks; the coordinate of each block is given by (blockIdx.x ,blockIdx.y )
  int GridDimX, GridDimY;

  // number of threads; the coordinate of each thread is given by (threadIdx.x,threadIdx.y)
  int BlockDimX, BlockDimY;

  // save output results in files
  double* Results;
  FILE *fp;

  Results = new double[IMAGE_SIZE * IMAGE_SIZE * 3];

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  BlockDimX = 100;
  BlockDimY = 1;
  GridDimX  = 1;
  GridDimY  = 50;
  sycl::range<2> gws (GridDimY*BlockDimY, GridDimX*BlockDimX);
  sycl::range<2> lws (BlockDimY, BlockDimX);

  //compute number of grides, to cover the whole image plane
  ImaDimX = (int)ceil((double)IMAGE_SIZE / (BlockDimX * GridDimX));
  ImaDimY = (int)ceil((double)IMAGE_SIZE / (BlockDimY * GridDimY));

  double *d_ResultsPixel = sycl::malloc_device<double>(IMAGE_SIZE * IMAGE_SIZE * 3, q);
  double *d_VariablesIn = sycl::malloc_device<double>(VarINNUM, q);

  q.memcpy(d_VariablesIn, VariablesIn, sizeof(double) * VarINNUM).wait();

  auto start = std::chrono::steady_clock::now();

  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class k1>(
          sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          task1(item, d_ResultsPixel, d_VariablesIn, GridIdxX, GridIdxY);
        });
      });
    }
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (task1) %f (s)\n", time * 1e-9f);

  q.memcpy(Results, d_ResultsPixel, sizeof(double) * IMAGE_SIZE * IMAGE_SIZE * 3).wait();

  //save result to output
  fp=fopen("Output_task1.txt","w");  
  if (fp != NULL) {
    fprintf(fp,"###output data:(alpha,  beta,  redshift)\n");

    for(int j = 0; j < IMAGE_SIZE; j++)
      for(int i = 0; i < IMAGE_SIZE; i++)
      {
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 0]);
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 1]);
        fprintf(fp, "%f\n", (float)Results[3 * (IMAGE_SIZE * j + i) + 2]);
      }
    fclose(fp);
  }

  A           = 0.;    // black hole spin
  INCLINATION = 45.;   // inclination angle in unit of degree                    
  SIZE        = IMAGE_SIZE;
  freq_obs    = 340e9; // observed frequency
  printf("task2: image size = %d  x  %d  pixels\n",IMAGE_SIZE,IMAGE_SIZE);

  q.memcpy(d_VariablesIn, VariablesIn, sizeof(double) * VarINNUM);

  double *d_K2_tab = sycl::malloc_device<double>(50, q);
  q.memcpy(d_K2_tab, K2_tab, sizeof(double) * 50);

  q.wait();
  start = std::chrono::steady_clock::now();

  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class k2>(
          sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          task2(item, d_ResultsPixel, d_VariablesIn, d_K2_tab, GridIdxX, GridIdxY);
        });
      });
    }
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (task2) %f (s)\n", time * 1e-9f);

  q.memcpy(Results, d_ResultsPixel, sizeof(double) * IMAGE_SIZE * IMAGE_SIZE * 3).wait();

  sycl::free(d_ResultsPixel, q);
  sycl::free(d_VariablesIn, q);
  sycl::free(d_K2_tab, q);

  fp=fopen("Output_task2.txt","w");  
  if (fp != NULL) {
    fprintf(fp,"###output data:(alpha,  beta, Luminosity (erg/sec))\n");

    for(int j = 0; j < IMAGE_SIZE; j++)
      for(int i = 0; i < IMAGE_SIZE; i++)
      {
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 0]);
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 1]);
        fprintf(fp, "%f\n", (float)Results[3 * (IMAGE_SIZE * j + i) + 2]);
      }
    fclose(fp);
  }

  delete [] Results;
  return 0;
}
