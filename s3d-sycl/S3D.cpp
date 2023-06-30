#include <cassert>
#include <chrono>
#include <string>
#include <sstream>
#include "OptionParser.h"
#include <sycl/sycl.hpp>
#include "S3D.h"

using namespace std;

// Forward declaration
template <class real>
void RunTest(string testName, sycl::queue &q, OptionParser &op);

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
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto t1 = std::chrono::high_resolution_clock::now();
  RunTest<float>("S3D-SP", q, op);
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time * 1e-9);

  t1 = std::chrono::high_resolution_clock::now();
  RunTest<double>("S3D-DP", q, op);
  t2 = std::chrono::high_resolution_clock::now();
  total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time * 1e-9);
}


template <class T> class ratt; 
template <class T> class rdsmh; 
template <class T> class gr_base; 
template <class T> class ratt2; 
template <class T> class ratt3; 
template <class T> class ratt4; 
template <class T> class ratt5; 
template <class T> class ratt6; 
template <class T> class ratt7; 
template <class T> class ratt8; 
template <class T> class ratt9; 
template <class T> class ratt10; 
template <class T> class ratx; 
template <class T> class ratxb; 
template <class T> class ratx2; 
template <class T> class ratx4; 
template <class T> class qssa; 
template <class T> class qssab; 
template <class T> class qssa2; 
template <class T> class rdwdot; 
template <class T> class rdwdot2; 
template <class T> class rdwdot3; 
template <class T> class rdwdot6; 
template <class T> class rdwdot7; 
template <class T> class rdwdot8; 
template <class T> class rdwdot9; 
template <class T> class rdwdot10; 

template <class real>
void RunTest(string testName, sycl::queue &q, OptionParser &op)
{
  // Number of grid points (specified in header file)
  int probSizes_SP[4] = { 8, 16, 32, 64};
  int probSizes_DP[4] = { 8, 16, 32, 64};
  int *probSizes = (sizeof(real) == sizeof(double)) ? probSizes_DP : probSizes_SP;
  int sizeClass = op.getOptionInt("size") - 1;
  assert(sizeClass >= 0 && sizeClass < 4);
  sizeClass = probSizes[sizeClass];
  int n = sizeClass * sizeClass * sizeClass;

  // Host variables
  real* host_t = (real*) malloc (n*sizeof(real));
  real* host_p = (real*) malloc (n*sizeof(real));
  real* host_y = (real*) malloc (Y_SIZE*n*sizeof(real));
  real* host_wdot = (real*) malloc (WDOT_SIZE*n*sizeof(real));
  real* host_molwt = (real*) malloc (WDOT_SIZE*n*sizeof(real));

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

  // Get kernel launch config, assuming n is divisible by block size
  sycl::range<1> lws(BLOCK_SIZE);
  sycl::range<1> gws (n);
  sycl::range<1> lws2(BLOCK_SIZE2);
  sycl::range<1> gws2 (n);

  real *d_t = sycl::malloc_device<real>(n, q);
  q.memcpy(d_t, host_t, n*sizeof(real));

  real *d_p = sycl::malloc_device<real>(n, q);
  q.memcpy(d_p, host_p, n*sizeof(real));

  real *d_y = sycl::malloc_device<real>(Y_SIZE*n, q);
  q.memcpy(d_y, host_y, Y_SIZE*n*sizeof(real));

  real *d_molwt = sycl::malloc_device<real>(WDOT_SIZE, q);
  q.memcpy(d_molwt, host_molwt, WDOT_SIZE*sizeof(real));

  real *d_wdot = sycl::malloc_device<real>(WDOT_SIZE*n, q);
  real *d_rf = sycl::malloc_device<real>(RF_SIZE*n, q);
  real *d_rb = sycl::malloc_device<real>(RB_SIZE*n, q);
  real *d_rklow = sycl::malloc_device<real>(RKLOW_SIZE*n, q);
  real *d_c = sycl::malloc_device<real>(C_SIZE*n, q);
  real *d_a = sycl::malloc_device<real>(A_SIZE*n, q);
  real *d_eg = sycl::malloc_device<real>(EG_SIZE*n, q);

  unsigned int passes = op.getOptionInt("passes");

  q.wait();
  auto start  = std::chrono::high_resolution_clock::now();

  for (unsigned int i = 0; i < passes; i++)
  {
    //  ratt_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt.sycl"
      });
    });

    //rdsmh_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdsmh<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdsmh.sycl"
      });
    });

    // gr_base <<< dim3(blks2), dim3(thrds2), 0, s2 >>> ( gpu_p, gpu_t, gpu_y, gpu_c, tconv, pconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class gr_base<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "gr_base.sycl"
      });
    });

    //  ratt2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt2<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt2.sycl"
      });
    });


    //ratt3_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt3<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt3.sycl"
      });
    });

    //ratt4_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt4<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt4.sycl"
      });
    });

    //ratt5_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt5<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt5.sycl"
      });
    });

    //  ratt6_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt6<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt6.sycl"
      });
    });
    //  ratt7_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt7<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt7.sycl"
      });
    });
    //ratt8_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt8<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt8.sycl"
      });
    });
    //ratt9_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt9<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt9.sycl"
      });
    });
    //ratt10_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rklow, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratt10<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratt10.sycl"
      });
    });

    //ratx_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratx<real>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
#include "ratx.sycl"
      });
    });
    //ratxb_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratxb<real>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
#include "ratxb.sycl"
      });
    });

    //ratx2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_c, gpu_rf, gpu_rb);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratx2<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratx2.sycl"
      });
    });
    //ratx4_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_c, gpu_rf, gpu_rb);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ratx4<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "ratx4.sycl"
      });
    });

    //qssa_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class qssa<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "qssa.sycl"
      });
    });

    //qssab_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class qssab<real>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
#include "qssab.sycl"
      });
    });
    //qssa2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class qssa2<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "qssa2.sycl"
      });
    });

    //  rdwdot_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot.sycl"
      });
    });

    //  rdwdot2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot2<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot2.sycl"
      });
    });
    //rdwdot3_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot3<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot3.sycl"
      });
    });

    //rdwdot6_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot6<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot6.sycl"
      });
    });
    //  rdwdot7_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot7<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot7.sycl"
      });
    });
    //rdwdot8_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot8<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot8.sycl"
      });
    });
    //  rdwdot9_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot9<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot9.sycl"
      });
    });
    // rdwdot10_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rdwdot10<real>>(sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
#include "rdwdot10.sycl"
      });
    });

    // Approximately 10k flops per grid point (estimated by Ramanan)
  }

  q.wait();
  auto end  = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nAverage time of executing s3d kernels: %lf (us)\n", (time * 1e-3) / passes);

  q.memcpy(host_wdot, d_wdot, WDOT_SIZE * n * sizeof(real)).wait();

  sycl::free(d_t, q);
  sycl::free(d_p, q);
  sycl::free(d_y, q);
  sycl::free(d_wdot, q);
  sycl::free(d_rf, q);
  sycl::free(d_rb, q);
  sycl::free(d_c, q);
  sycl::free(d_rklow, q);
  sycl::free(d_a, q);
  sycl::free(d_eg, q);
  sycl::free(d_molwt, q);

  // Print out answers for verification
  for (int i=0; i<WDOT_SIZE; i++) {
      printf("% 23.16E ", host_wdot[i*n]);
      if (i % 3 == 2)
          printf("\n");
  }
  printf("\n");

  free(host_t);
  free(host_p);
  free(host_y);
  free(host_wdot);
  free(host_molwt);
}
