/*
* Portions Copyright (c) 1993-2015 NVIDIA Corporation.  All rights reserved.
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
* Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
* Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
*
* Sobol Quasi-random Number Generator example
*
* Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
* http://people.maths.ox.ac.uk/~gilesm/
*
* and C code developed by Stephen Joe, University of Waikato, New Zealand
* and Frances Kuo, University of New South Wales, Australia
* http://web.maths.unsw.edu.au/~fkuo/sobol/
*
* For theoretical background see:
*
* P. Bratley and B.L. Fox.
* Implementing Sobol's quasirandom sequence generator
* http://portal.acm.org/citation.cfm?id=42288
* ACM Trans. on Math. Software, 14(1):88-100, 1988
*
* S. Joe and F. Kuo.
* Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
* http://portal.acm.org/citation.cfm?id=641879
* ACM Trans. on Math. Software, 29(1):49-57, 2003
*
*/

#include "sobol.h"
#include "sobol_gpu.h"

#define k_2powneg32 2.3283064E-10F


#pragma omp declare target
int _ffs(const int x) {
  for (int i = 0; i < 32; i++)
    if ((x >> i) & 1) return (i+1);
  return 0;
};
#pragma omp end declare target

double sobolGPU(int repeat, int n_vectors, int n_dimensions, 
                unsigned int *dir, float *out)
{
    const int threadsperblock = 64;

    // This implementation of the generator outputs all the draws for
    // one dimension in a contiguous region of memory, followed by the
    // next dimension and so on.
    // Therefore all threads within a block will be processing different
    // vectors from the same dimension. As a result we want the total
    // number of blocks to be a multiple of the number of dimensions.
    size_t dimGrid_y = n_dimensions;
    size_t dimGrid_x;

    // If the number of dimensions is large then we will set the number
    // of blocks to equal the number of dimensions (i.e. dimGrid.x = 1)
    // but if the number of dimensions is small (e.g. less than four per
    // multiprocessor) then we'll partition the vectors across blocks
    // (as well as threads).
    if (n_dimensions < (4 * 24))
    {
        dimGrid_x = 4 * 24;
    }
    else
    {
        dimGrid_x = 1;
    }

    // Cap the dimGrid.x if the number of vectors is small
    if (dimGrid_x > (unsigned int)(n_vectors / threadsperblock))
    {
        dimGrid_x = (n_vectors + threadsperblock - 1) / threadsperblock;
    }

    // Round up to a power of two, required for the algorithm so that
    // stride is a power of two.
    unsigned int targetDimGridX = dimGrid_x;

    for (dimGrid_x = 1 ; dimGrid_x < targetDimGridX ; dimGrid_x *= 2);

    // Fix the number of threads
    size_t numTeam =  dimGrid_x * dimGrid_y;

    auto start = std::chrono::steady_clock::now();

    // Execute GPU kernel
    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams num_teams(numTeam) thread_limit(threadsperblock)
      {
	unsigned int v[n_directions];
        #pragma omp parallel 
	{
	  unsigned int teamX = omp_get_team_num() % dimGrid_x;
	  unsigned int teamY = omp_get_team_num() / dimGrid_x; 
	  unsigned int tidX = omp_get_thread_num();
	  unsigned int threadSizeX = omp_get_num_threads();

          dir += n_directions * teamY;
          out += n_vectors * teamY;

          // Copy the direction numbers for this dimension into shared
          // memory - there are only 32 direction numbers so only the
          // first 32 (n_directions) threads need participate.
          if (tidX < n_directions)
          {
            v[tidX] = dir[tidX];
          }

          #pragma omp barrier

          // Set initial index (i.e. which vector this thread is
          // computing first) and stride (i.e. step to the next vector
          // for this thread)
          int i0     = teamX * threadSizeX + tidX;
          int stride = dimGrid_x * threadSizeX;

          // Get the gray code of the index
          // c.f. Numerical Recipes in C, chapter 20
          // http://www.nrbook.com/a/bookcpdf/c20-2.pdf
          unsigned int g = i0 ^ (i0 >> 1);

          // Initialisation for first point x[i0]
          // In the Bratley and Fox paper this is equation (*), where
          // we are computing the value for x[n] without knowing the
          // value of x[n-1].
          unsigned int X = 0;
          unsigned int mask;

          for (unsigned int k = 0 ; k < _ffs(stride) - 1 ; k++)
          {
              // We want X ^= g_k * v[k], where g_k is one or zero.
              // We do this by setting a mask with all bits equal to
              // g_k. In reality we keep shifting g so that g_k is the
              // LSB of g. This way we avoid multiplication.
              mask = - (g & 1);
              X ^= mask & v[k];
              g = g >> 1;
          }

          if (i0 < n_vectors)
          {
              out[i0] = (float)X * k_2powneg32;
          }

          // Now do rest of points, using the stride
          // Here we want to generate x[i] from x[i-stride] where we
          // don't have any of the x in between, therefore we have to
          // revisit the equation (**), this is easiest with an example
          // so assume stride is 16.
          // From x[n] to x[n+16] there will be:
          //   8 changes in the first bit
          //   4 changes in the second bit
          //   2 changes in the third bit
          //   1 change in the fourth
          //   1 change in one of the remaining bits
          //
          // What this means is that in the equation:
          //   x[n+1] = x[n] ^ v[p]
          //   x[n+2] = x[n+1] ^ v[q] = x[n] ^ v[p] ^ v[q]
          //   ...
          // We will apply xor with v[1] eight times, v[2] four times,
          // v[3] twice, v[4] once and one other direction number once.
          // Since two xors cancel out, we can skip even applications
          // and just apply xor with v[4] (i.e. log2(16)) and with
          // the current applicable direction number.
          // Note that all these indices count from 1, so we need to
          // subtract 1 from them all to account for C arrays counting
          // from zero.
          unsigned int v_log2stridem1 = v[_ffs(stride) - 2];
          unsigned int v_stridemask = stride - 1;

          for (unsigned int i = i0 + stride ; i < n_vectors ; i += stride)
          {
              // x[i] = x[i-stride] ^ v[b] ^ v[c]
              //  where b is log2(stride) minus 1 for C array indexing
              //  where c is the index of the rightmost zero bit in i,
              //  not including the bottom log2(stride) bits, minus 1
              //  for C array indexing
              // In the Bratley and Fox paper this is equation (**)
              X ^= v_log2stridem1 ^ v[_ffs(~((i - stride) | v_stridemask)) - 1];
              out[i] = (float)X * k_2powneg32;
          }

        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return time;
}
