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
#include "common.h"
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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  BlockDimX = 100;
  BlockDimY = 1;
  GridDimX  = 1;
  GridDimY  = 50;
  range<2> gws (GridDimY*BlockDimY, GridDimX*BlockDimX);
  range<2> lws (BlockDimY, BlockDimX);

  //compute number of grides, to cover the whole image plane
  ImaDimX = (int)ceil((double)IMAGE_SIZE / (BlockDimX * GridDimX));
  ImaDimY = (int)ceil((double)IMAGE_SIZE / (BlockDimY * GridDimY));

  buffer<double, 1> d_ResultsPixel (IMAGE_SIZE * IMAGE_SIZE * 3);
  buffer<double, 1> d_VariablesIn (VarINNUM);

  q.submit([&] (handler &cgh) {
    auto acc = d_VariablesIn.get_access<sycl_discard_write>(cgh);
    cgh.copy(VariablesIn, acc);
  });

  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      q.submit([&] (handler &cgh) {
        auto result = d_ResultsPixel.get_access<sycl_discard_write>(cgh);
        auto input = d_VariablesIn.get_access<sycl_read>(cgh);
        cgh.parallel_for<class k1>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          task1(item, result.get_pointer(), input.get_pointer(), GridIdxX, GridIdxY);
        });
      });
    }
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_ResultsPixel.get_access<sycl_read>(cgh);
    cgh.copy(acc, Results);
  }).wait();

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

  q.submit([&] (handler &cgh) {
    auto acc = d_VariablesIn.get_access<sycl_discard_write>(cgh);
    cgh.copy(VariablesIn, acc);
  });

  buffer<const double, 1> d_K2_tab (K2_tab, 50); 
  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      q.submit([&] (handler &cgh) {
        auto result = d_ResultsPixel.get_access<sycl_discard_write>(cgh);
        auto input = d_VariablesIn.get_access<sycl_read>(cgh);
        auto table = d_K2_tab.get_access<sycl_read, sycl_cmem>(cgh);
        cgh.parallel_for<class k2>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          task2(item, result.get_pointer(), input.get_pointer(), table.get_pointer(), GridIdxX, GridIdxY);
        });
      });
    }
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_ResultsPixel.get_access<sycl_read>(cgh);
    cgh.copy(acc, Results);
  }).wait();

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
