#include <cassert>
#include <chrono>
#include <string>
#include <sstream>
#include "OptionParser.h"
#include "S3D.h"

using namespace std;

// Forward declaration
template <class real>
void RunTest(string testName, OptionParser &op);

// ********************************************************
// Function: toString
//
// Purpose:
//   Simple templated function to convert objects into
//   strings using stringstream
//
// Arguments:
//   t: the object to convert to a string
//
// Returns:  a string representation
//
// Modifications:
//
// ********************************************************
template<class T> inline string toString(const T& t)
{
  stringstream ss;
  ss << t;
  return ss.str();
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
  void
addBenchmarkSpecOptions(OptionParser &op)
{
  ; // No S3D specific options
}

void RunBenchmark(OptionParser &op)
{
  auto t1 = std::chrono::high_resolution_clock::now();
  RunTest<float>("S3D-SP", op);
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time * 1e-9);

  t1 = std::chrono::high_resolution_clock::now();
  RunTest<float>("S3D-DP", op);
  t2 = std::chrono::high_resolution_clock::now();
  total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time * 1e-9);
}

  template <class real>
void RunTest(string testName, OptionParser &op)
{
  // Number of grid points (specified in header file)
  int probSizes_SP[4] = { 8, 16, 32, 64 };
  int probSizes_DP[4] = { 8, 16, 32, 64 };
  int *probSizes = (sizeof(real) == sizeof(double)) ? probSizes_DP : probSizes_SP;
  int sizeClass = op.getOptionInt("size") - 1;
  assert(sizeClass >= 0 && sizeClass < 4);
  sizeClass = probSizes[sizeClass];
  int n = sizeClass * sizeClass * sizeClass;

  // Host variables
  real* host_t = (real*) malloc (n*sizeof(real));
  real* host_p = (real*) malloc (n*sizeof(real));
  real* host_y = (real*) malloc (Y_SIZE*n*sizeof(real));
  real* host_molwt = (real*) malloc (WDOT_SIZE*sizeof(real));

  real* RF = (real*) malloc (RF_SIZE*n*sizeof(real));
  real* RB = (real*) malloc (RB_SIZE*n*sizeof(real));
  real* RKLOW = (real*) malloc (RKLOW_SIZE*n*sizeof(real));
  real* C = (real*) malloc (C_SIZE*n*sizeof(real));
  real* A = (real*) malloc (A_SIZE*n*sizeof(real));
  real* EG = (real*) malloc (EG_SIZE*n*sizeof(real));
  real* WDOT = (real*) malloc (WDOT_SIZE*n*sizeof(real));

  // Initialize Test Problem

  // For now these are just 1, to compare results between cpu & gpu
  real rateconv = 1.0;
  real tconv = 1.0;
  real pconv = 1.0;

  // Initialize temp and pressure
  for (int i=0; i<n; i++)
  {
    host_p[i] = 1.0132e6;
    host_t[i] = 1000.0;
  }

  // Init molwt: for now these are just 1, to compare results betw. cpu & gpu
  for (int i=0; i<WDOT_SIZE; i++)
  {
    host_molwt[i] = 1;
  }

  // Initialize mass fractions
  for (int j=0; j<Y_SIZE; j++)
  {
    for (int i=0; i<n; i++)
    {
      host_y[(j*n)+i]= 0.0;
      if (j==14)
        host_y[(j*n)+i] = 0.064;
      if (j==3)
        host_y[(j*n)+i] = 0.218;
      if (j==21)
        host_y[(j*n)+i] = 0.718;
    }
  }

  real *T = host_t;
  real *P = host_p;
  real *Y = host_y;
  real *molwt = host_molwt;

  int thrds = BLOCK_SIZE;
  int thrds2 = BLOCK_SIZE2;

  unsigned int passes = op.getOptionInt("passes");

#pragma omp target data map(to: T[0:n], P[0:n], Y[0:Y_SIZE*n], molwt[0:WDOT_SIZE]) \
                        map(alloc:RF[0:RF_SIZE*n], \
                                  RB[0:RB_SIZE*n], \
                                  RKLOW[0:RKLOW_SIZE*n], \
                                  C[0:C_SIZE*n], \
                                  A[0:A_SIZE*n], \
                                  EG[0:EG_SIZE*n]) \
                        map(from: WDOT[0:WDOT_SIZE*n])
  {
    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < passes; i++)
    {
      //  ratt_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt.h"
      }

      //rdsmh_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "rdsmh.h"
      }

      // gr_base <<< dim3(blks2), dim3(thrds2), 0, s2 >>> ( gpu_p, gpu_t, gpu_y, gpu_c, tconv, pconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "gr_base.h"
      }

      //  ratt2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt2.h"
      }

      //ratt3_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt3.h"
      }

      //ratt4_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt4.h"
      }

      //ratt5_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt5.h"
      }

#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt6.h"
      }
      //  ratt7_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt7.h"
      }
      //ratt8_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt8.h"
      }
      //ratt9_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt9.h"
      }
      //ratt10_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rklow, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds2)
      for (int i = 0; i < n; i++) {
#include "ratt10.h"
      }

      //ratx_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds)
      for (int i = 0; i < n; i++) {
#include "ratx.h"
      }

      //ratxb_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv);
#pragma omp target teams distribute parallel for thread_limit(thrds)
      for (int i = 0; i < n; i++) {
#include "ratxb.h"
      }

      //ratx2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_c, gpu_rf, gpu_rb);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "ratx2.h"
        }
      }

      //ratx4_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_c, gpu_rf, gpu_rb);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "ratx4.h"
        }
      }

      //qssa_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "qssa.h"
        }
      }

      //qssab_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "qssab.h"
        }
      }
      //qssa2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "qssa2.h"
        }
      }

      //  rdwdot_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot.h"
        }
      }

      //  rdwdot2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot2.h"
        }
      }

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot3.h"
        }
      }

      //rdwdot6_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot6.h"
        }
      }

      //  rdwdot7_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot7.h"
        }
      }

      //rdwdot8_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot8.h"
        }
      }

      //  rdwdot9_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot9.h"
        }
      }

      // rdwdot10_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
      {
#pragma omp parallel
        {
#include "rdwdot10.h"
        }
      }
      // Approximately 10k flops per grid point (estimated by Ramanan)
    }
    auto end  = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("\nAverage time of executing s3d kernels: %lf (us)\n", (time * 1e-3) / passes);
  }

  // Print out answers for verification
  for (int i=0; i<WDOT_SIZE; i++) {
    printf("% 23.16E ", WDOT[i*n]);
    if (i % 3 == 2)
      printf("\n");
  }
  printf("\n");

  free(host_t);
  free(host_p);
  free(host_y);
  free(host_molwt);
  free(RF);
  free(RB);
  free(RKLOW);
  free(C);
  free(A);
  free(EG);
  free(WDOT);
}
