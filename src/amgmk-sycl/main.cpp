/*BHEADER****************************************************************
 * (c) 2007   The Regents of the University of California               *
 *                                                                      *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright       *
 * notice and disclaimer.                                               *
 *                                                                      *
 *EHEADER****************************************************************/

//--------------
//  A micro kernel 
//--------------
#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>

#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
#endif

#include "headers.h"

// CUDA/HIP block size or OpenCL work-group size
#define BLOCK_SIZE 256

// 
const int testIter   = 500;
double totalWallTime = 0.0;

// 
void test_Matvec();
void test_Relax();
void test_Axpy();

//
int main(int argc, char *argv[])
{
#ifdef _OPENMP
  double t0        = 0.0,
         t1        = 0.0,
#else
  printf("**** Warning: OpenMP is disabled ****\n");
#endif

  double del_wtime = 0.0;

#ifdef _OPENMP
  int  max_num_threads;
#endif


  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//  CORAL  AMGmk Benchmark Version 1.0 \n");
  printf("// \n");
  printf("//------------ \n");

  printf("\n testIter   = %d \n\n", testIter );  

 
#ifdef _OPENMP
  printf("\n testIter   = %d \n\n", testIter );  
  #pragma omp parallel
     #pragma omp master
        max_num_threads = omp_get_num_threads();
   printf("\nmax_num_threads = %d \n\n",max_num_threads );
#endif


#ifdef _OPENMP
  t0 = omp_get_wtime();
#else
  auto t0 = std::chrono::steady_clock::now();
#endif

  // Matvec
  totalWallTime = 0.0;
 
  test_Matvec();

  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//   MATVEC\n");
  printf("// \n");
  printf("//------------ \n");

  printf("\nWall time = %f seconds. \n", totalWallTime);


  // Relax
  totalWallTime = 0.0;

  test_Relax();

  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//   Relax\n");
  printf("// \n");
  printf("//------------ \n");

  printf("\nTotal kernel time = %f seconds. \n", totalWallTime);


  // Axpy
  totalWallTime = 0.0;
 
  test_Axpy();

  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//   Axpy\n");
  printf("// \n");
  printf("//------------ \n");

  printf("\nWall time = %f seconds. \n", totalWallTime);

#ifdef _OPENMP
  t1 = omp_get_wtime();
  del_wtime = t1 - t0;
#else
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = t1 - t0;
  del_wtime = diff.count();
#endif


  printf("\nTotal Wall time = %f seconds. \n", del_wtime);

  return  0;
}

void test_Matvec()
{
#ifdef _OPENMP
  double t0 = 0.0,
         t1 = 0.0;
#endif

  hypre_CSRMatrix *A;
  hypre_Vector *x, *y, *sol;
  int nx, ny, nz, i;
  double *values;
  double *y_data, *sol_data;
  double error, diff;

  nx = 50;  /* size per proc nx*ny*nz */
  ny = 50;
  nz = 50;

  values = hypre_CTAlloc(double, 4);
  values[0] = 6; 
  values[1] = -1;
  values[2] = -1;
  values[3] = -1;

  A = GenerateSeqLaplacian(nx, ny, nz, values, &y, &x, &sol);

  hypre_SeqVectorSetConstantValues(x,1);
  hypre_SeqVectorSetConstantValues(y,0);

#ifdef _OPENMP
  t0 = omp_get_wtime();
#else
  auto t0 = std::chrono::steady_clock::now();
#endif

  for (i=0; i<testIter; ++i)
      hypre_CSRMatrixMatvec(1,A,x,0,y);

#ifdef _OPENMP
  t1 = omp_get_wtime() ;
  totalWallTime += t1 - t0;
#else
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> tdiff = t1 - t0;
  totalWallTime += tdiff.count();
#endif

 
  y_data = hypre_VectorData(y);
  sol_data = hypre_VectorData(sol);

  error = 0;
  for (i=0; i < nx*ny*nz; i++)
  {
      diff = std::fabs(y_data[i]-sol_data[i]);
      if (diff > error) error = diff;
  }
     
  if (error > 0) printf(" \n Matvec: error: %e\n", error);

  hypre_TFree(values);
  hypre_CSRMatrixDestroy(A);
  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
  hypre_SeqVectorDestroy(sol);

}

void test_Relax()
{
#ifdef _OPENMP
  double t0 = 0.0,
         t1 = 0.0;
#endif

  hypre_CSRMatrix *A;
  hypre_Vector *x, *y, *sol;
  int nx, ny, nz, i;
  double *values;
  double diff, error;

  nx = 50;  /* size per proc nx*ny*nz */
  ny = 50;
  nz = 50;

  values = hypre_CTAlloc(double, 4);
  values[0] = 6; 
  values[1] = -1;
  values[2] = -1;
  values[3] = -1;

  A = GenerateSeqLaplacian(nx, ny, nz, values, &y, &x, &sol);

  hypre_SeqVectorSetConstantValues(x,1);

  double         *A_diag_data  = hypre_CSRMatrixData(A);
  int            *A_diag_i     = hypre_CSRMatrixI(A);
  int            *A_diag_j     = hypre_CSRMatrixJ(A);

  int             n       = hypre_CSRMatrixNumRows(A);
  int             nonzero = hypre_CSRMatrixNumNonzeros(A);

  double         *u_data  = hypre_VectorData(x);
  //int         u_data_size  = hypre_VectorSize(x);

  double         *f_data  = hypre_VectorData(sol);
  //int         f_data_size  = hypre_VectorSize(sol);

  int             grid_size = nx*ny*nz;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *d_A_diag_data = sycl::malloc_device<double>(nonzero, q);
  q.memcpy(d_A_diag_data, A_diag_data, sizeof(double)*nonzero);

     int *d_A_diag_i = sycl::malloc_device<int>(grid_size+1, q);
  q.memcpy(d_A_diag_i, A_diag_i, sizeof(int)*(grid_size+1));

     int *d_A_diag_j = sycl::malloc_device<int>(nonzero, q); 
  q.memcpy(d_A_diag_j, A_diag_j, sizeof(int)*nonzero);

  double *d_u_data = sycl::malloc_device<double>(grid_size, q); 
  q.memcpy(d_u_data, u_data, sizeof(double)*grid_size);

  double *d_f_data = sycl::malloc_device<double>(grid_size, q); 
  q.memcpy(d_f_data, f_data, sizeof(double)*grid_size);

  q.wait();
  sycl::range<1> lws (BLOCK_SIZE);
  sycl::range<1> gws ((n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);

#ifdef _OPENMP
  t0 = omp_get_wtime();
#else
  auto t0 = std::chrono::steady_clock::now();
#endif

  for (i = 0; i < testIter; ++i) {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class relax>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i >= n) return;
          
          /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/
          
          if ( d_A_diag_data[d_A_diag_i[i]] != 0.0)
          {
            double res = d_f_data[i];
            for (int jj = d_A_diag_i[i]+1; jj < d_A_diag_i[i+1]; jj++)
            {
              int ii = d_A_diag_j[jj];
              res -= d_A_diag_data[jj] * d_u_data[ii];
            }
            d_u_data[i] = res / d_A_diag_data[d_A_diag_i[i]];
          }
       });
    });
  } // for

  q.wait();

#ifdef _OPENMP
  t1 = omp_get_wtime();
  totalWallTime += t1 - t0;
#else
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> tdiff = t1 - t0;
  totalWallTime += tdiff.count();
#endif

  q.memcpy(u_data, d_u_data, sizeof(double)*grid_size).wait();

  sycl::free(d_A_diag_data, q);
  sycl::free(d_A_diag_i, q);
  sycl::free(d_A_diag_j, q);
  sycl::free(d_u_data, q);
  sycl::free(d_f_data, q);

  error = 0;
  for (i=0; i < nx*ny*nz; i++)
  {
      diff = std::fabs(u_data[i]-1);
      if (diff > error) error = diff;
  }
     
  if (error > 0) printf(" \n Relax: error: %e\n", error);

  hypre_TFree(values);
  hypre_CSRMatrixDestroy(A);
  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
  hypre_SeqVectorDestroy(sol);

}

void test_Axpy()
{
#ifdef _OPENMP
  double t0 = 0.0,
         t1 = 0.0;
#endif


  hypre_Vector *x, *y;
  int nx, i;
  double alpha=0.5;
  double diff, error;
  double *y_data;

  nx = 125000;  /* size per proc  */

  x = hypre_SeqVectorCreate(nx);
  y = hypre_SeqVectorCreate(nx);

  hypre_SeqVectorInitialize(x);
  hypre_SeqVectorInitialize(y);

  hypre_SeqVectorSetConstantValues(x,1);
  hypre_SeqVectorSetConstantValues(y,1);

 
#ifdef _OPENMP
  t0 = omp_get_wtime();
#else
  auto t0 = std::chrono::steady_clock::now();
#endif

  for (i=0; i<testIter; ++i)
      hypre_SeqVectorAxpy(alpha,x,y);
#ifdef _OPENMP
  t1 = omp_get_wtime();
#else
  auto t1 = std::chrono::steady_clock::now();
#endif
  

  y_data = hypre_VectorData(y);
  error = 0;
  for (i=0; i < nx; i++)
  {
    diff = std::fabs(y_data[i]-1-0.5*(double)testIter);
      if (diff > error) error = diff;
  }
     
  if (error > 0) printf(" \n Axpy: error: %e\n", error);

#ifdef _OPENMP
  totalWallTime += t1 - t0; 
#else
  std::chrono::duration<double> tdiff = t1 - t0;
  totalWallTime += tdiff.count();
#endif

  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);

}
