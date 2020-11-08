#include <cassert>
#include <string>
#include <sstream>
#include "OptionParser.h"
#include "common.h"
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
  // Always run the single precision test
  RunTest<float>("S3D-SP", op);
  //RunTest<float>("S3D-DP", op);
}

  template <class real>
void RunTest(string testName, OptionParser &op)
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

  { // sycl scope

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // Get kernel launch config, assuming n is divisible by block size
  range<1> lws(BLOCK_SIZE);
  range<1> gws (n);
  range<1> lws2(BLOCK_SIZE2);
  range<1> gws2 (n);

  buffer<real, 1> gpu_t (host_t, n);
  buffer<real, 1> gpu_p (host_p, n);
  buffer<real, 1> gpu_y (host_y, Y_SIZE*n);
  buffer<real, 1> gpu_molwt (host_molwt, WDOT_SIZE);
  buffer<real, 1> gpu_wdot (host_wdot, WDOT_SIZE*n);

  buffer<real, 1> gpu_rf (RF_SIZE*n);
  buffer<real, 1> gpu_rb (RB_SIZE*n);
  buffer<real, 1> gpu_rklow (RKLOW_SIZE*n);
  buffer<real, 1> gpu_c (C_SIZE*n);
  buffer<real, 1> gpu_a (A_SIZE*n);
  buffer<real, 1> gpu_eg (EG_SIZE*n);

  unsigned int passes = op.getOptionInt("passes");
  for (unsigned int i = 0; i < passes; i++)
  {
    //  ratt_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt.sycl"
            });
        });

    //rdsmh_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_eg, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto EG = gpu_eg.template get_access<sycl_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdsmh.sycl"
            });
        });

    // gr_base <<< dim3(blks2), dim3(thrds2), 0, s2 >>> ( gpu_p, gpu_t, gpu_y, gpu_c, tconv, pconv);
    q.submit([&] (handler &cgh) {
        auto P = gpu_p.template get_access<sycl_read>(cgh);
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto Y = gpu_y.template get_access<sycl_read>(cgh);
        auto C = gpu_c.template get_access<sycl_read_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "gr_base.sycl"
            });
        });

    //  ratt2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt2.sycl"
            });
        });


    //ratt3_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt3.sycl"
            });
        });

    //ratt4_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);

    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt4.sycl"
            });
        });

    //ratt5_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);

    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt5.sycl"
            });
        });

    //  ratt6_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt6.sycl"
            });
        });
    //  ratt7_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt7.sycl"
            });
        });
    //ratt8_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt8.sycl"
            });
        });
    //ratt9_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RB = gpu_rb.template get_access<sycl_write>(cgh);
        auto EG = gpu_eg.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt9.sycl"
            });
        });
    //ratt10_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_t, gpu_rklow, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto RKLOW = gpu_rklow.template get_access<sycl_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratt10.sycl"
            });
        });

    //ratx_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto C = gpu_c.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read_write>(cgh);
        auto RB = gpu_rb.template get_access<sycl_read_write>(cgh);
        auto RKLOW = gpu_rklow.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
#include "ratx.sycl"
            });
        });
    //ratxb_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv);
    q.submit([&] (handler &cgh) {
        auto T = gpu_t.template get_access<sycl_read>(cgh);
        auto C = gpu_c.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read_write>(cgh);
        auto RB = gpu_rb.template get_access<sycl_read_write>(cgh);
        auto RKLOW = gpu_rklow.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
#include "ratxb.sycl"
            });
        });

    //ratx2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_c, gpu_rf, gpu_rb);
    q.submit([&] (handler &cgh) {
        auto C = gpu_c.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read_write>(cgh);
        //auto RB = gpu_rb.template get_access<sycl_read_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratx2.sycl"
            });
        });
    //ratx4_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_c, gpu_rf, gpu_rb);
    q.submit([&] (handler &cgh) {
        auto C = gpu_c.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read_write>(cgh);
        auto RB = gpu_rb.template get_access<sycl_read_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "ratx4.sycl"
            });
        });

    //qssa_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
    q.submit([&] (handler &cgh) {
        auto A = gpu_a.template get_access<sycl_write>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read_write>(cgh);
        auto RB = gpu_rb.template get_access<sycl_read_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "qssa.sycl"
            });
        });

    //qssab_kernel <<< dim3(blks), dim3(thrds), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
    q.submit([&] (handler &cgh) {
        auto A = gpu_a.template get_access<sycl_read_write>(cgh);
        //auto RF = gpu_rf.template get_access<sycl_read_write>(cgh);
        //auto RB = gpu_rb.template get_access<sycl_read_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
#include "qssab.sycl"
            });
        });
    //qssa2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_a);
    q.submit([&] (handler &cgh) {
        auto A = gpu_a.template get_access<sycl_read>(cgh);
        auto RF = gpu_rf.template get_access<sycl_read_write>(cgh);
        auto RB = gpu_rb.template get_access<sycl_read_write>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "qssa2.sycl"
            });
        });

    //  rdwdot_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot.sycl"
            });
        });

    //  rdwdot2_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot2.sycl"
            });
        });
    //rdwdot3_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot3.sycl"
            });
        });

    //rdwdot6_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot6.sycl"
            });
        });
    //  rdwdot7_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot7.sycl"
            });
        });
    //rdwdot8_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot8.sycl"
            });
        });
    //  rdwdot9_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot9.sycl"
            });
        });
    // rdwdot10_kernel <<< dim3(blks2), dim3(thrds2), 0, s1 >>> ( gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt);
    q.submit([&] (handler &cgh) {
        auto RKF = gpu_rf.template get_access<sycl_read>(cgh);
        auto RKR = gpu_rb.template get_access<sycl_read>(cgh);
        auto WDOT = gpu_wdot.template get_access<sycl_write>(cgh);
        auto molwt = gpu_molwt.template get_access<sycl_read>(cgh);
        cgh.parallel_for(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
#include "rdwdot10.sycl"
            });
        });

    // Approximately 10k flops per grid point (estimated by Ramanan)
  }
  q.wait();

  }

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
