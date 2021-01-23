#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cudacommon.h"
#include "OptionParser.h"
#include "S3D.h"
#include "gr_base.h"
#include "ratt.h"
#include "ratt2.h"
#include "ratx.h"
#include "qssa.h"
#include "qssa2.h"
#include "rdwdot.h"

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

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the S3D benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
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
void RunBenchmark(OptionParser &op)
{
    RunTest<float>("S3D-SP", op);
    RunTest<double>("S3D-DP", op);
}

template <class real> void RunTest(string testName, OptionParser &op) try {
   dpct::device_ext &dev_ct1 = dpct::get_current_device();
   sycl::queue &q_ct1 = dev_ct1.default_queue();
    // Number of grid points (specified in header file)
    int probSizes_SP[4] = { 8, 16, 32, 64};
    int probSizes_DP[4] = { 8, 16, 32, 64};
    int *probSizes = (sizeof(real) == sizeof(double)) ? probSizes_DP : probSizes_SP;
    int sizeClass = op.getOptionInt("size") - 1;
    assert(sizeClass >= 0 && sizeClass < 4);
    sizeClass = probSizes[sizeClass];
    int n = sizeClass * sizeClass * sizeClass;

    // Host variables
    real* host_t;
    real* host_p;
    real* host_y;
    real* host_wdot;
    real* host_molwt;

    // GPU Variables
    real* gpu_t; //Temperatures array
    real* gpu_p; //Pressures array
    real* gpu_y; //Mass fractions
    real* gpu_wdot; //Output variables

    // GPU Intermediate Variables
    real* gpu_rf, *gpu_rb;
    real* gpu_rklow;
    real* gpu_c;
    real* gpu_a;
    real* gpu_eg;
    real* gpu_molwt;

    // Malloc host memory
   /*
   DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (host_t = (real *)sycl::malloc_host(n * sizeof(real), q_ct1), 0));
   /*
   DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (host_p = (real *)sycl::malloc_host(n * sizeof(real), q_ct1), 0));
   /*
   DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (host_y = (real *)sycl::malloc_host(Y_SIZE * n * sizeof(real), q_ct1),
        0));
   /*
   DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((host_wdot = (real *)sycl::malloc_host(
                       WDOT_SIZE * n * sizeof(real), q_ct1),
                   0));
   /*
   DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (host_molwt = (real *)sycl::malloc_host(WDOT_SIZE * sizeof(real), q_ct1),
        0));

    // Initialize Test Problem

    // For now these are just 1 for verification
    real rateconv = 1.0;
    real tconv = 1.0;
    real pconv = 1.0;

    // Initialize temp and pressure
    for (int i=0; i<n; i++)
    {
        host_p[i] = 1.0132e6;
        host_t[i] = 1000.0;
    }

    // Init molwt: for now these are just 1 for verification
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

    // Malloc GPU memory
   /*
   DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_t = (real *)sycl::malloc_device(n * sizeof(real), q_ct1), 0));
   /*
   DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_p = (real *)sycl::malloc_device(n * sizeof(real), q_ct1), 0));
   /*
   DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_y = (real *)sycl::malloc_device(Y_SIZE * n * sizeof(real), q_ct1),
        0));
   /*
   DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((gpu_wdot = (real *)sycl::malloc_device(
                       WDOT_SIZE * n * sizeof(real), q_ct1),
                   0));
   /*
   DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_rf = (real *)sycl::malloc_device(RF_SIZE * n * sizeof(real), q_ct1),
        0));
   /*
   DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_rb = (real *)sycl::malloc_device(RB_SIZE * n * sizeof(real), q_ct1),
        0));
   /*
   DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((gpu_rklow = (real *)sycl::malloc_device(
                       RKLOW_SIZE * n * sizeof(real), q_ct1),
                   0));
   /*
   DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_c = (real *)sycl::malloc_device(C_SIZE * n * sizeof(real), q_ct1),
        0));
   /*
   DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_a = (real *)sycl::malloc_device(A_SIZE * n * sizeof(real), q_ct1),
        0));
   /*
   DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (gpu_eg = (real *)sycl::malloc_device(EG_SIZE * n * sizeof(real), q_ct1),
        0));
   /*
   DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((
       gpu_molwt = (real *)sycl::malloc_device(WDOT_SIZE * sizeof(real), q_ct1),
       0));

    // Get kernel launch config, assuming n is divisible by block size
   sycl::range<3> thrds(BLOCK_SIZE, 1, 1);
   sycl::range<3> blks(n / BLOCK_SIZE, 1, 1);
   sycl::range<3> thrds2(BLOCK_SIZE2, 1, 1);
   sycl::range<3> blks2(n / BLOCK_SIZE2, 1, 1);

    // Download of gpu_t, gpu_p, gpu_y, gpu_molwt
   /*
   DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((q_ct1.memcpy(gpu_t, host_t, n * sizeof(real)), 0));
   /*
   DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((q_ct1.memcpy(gpu_p, host_p, n * sizeof(real)), 0));
   /*
   DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((q_ct1.memcpy(gpu_y, host_y, Y_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (q_ct1.memcpy(gpu_molwt, host_molwt, WDOT_SIZE * sizeof(real)), 0));

    unsigned int passes = op.getOptionInt("passes");
    for (unsigned int i = 0; i < passes; i++)
    {
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt_kernel(gpu_t, gpu_rf, tconv, item_ct1);
             });
      });

      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdsmh_kernel(gpu_t, gpu_eg, tconv, item_ct1);
             });
      });

      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                gr_base(gpu_p, gpu_t, gpu_y, gpu_c, tconv, pconv, item_ct1);
             });
      });

      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt2_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt3_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt4_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt5_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt6_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt7_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt8_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt9_kernel(gpu_t, gpu_rf, gpu_rb, gpu_eg, tconv, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratt10_kernel(gpu_t, gpu_rklow, tconv, item_ct1);
             });
      });

      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range = sycl::range<3>(blks) * sycl::range<3>(thrds);
         auto dpct_local_range = sycl::range<3>(thrds);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratx_kernel(gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv,
                            item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range = sycl::range<3>(blks) * sycl::range<3>(thrds);
         auto dpct_local_range = sycl::range<3>(thrds);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratxb_kernel(gpu_t, gpu_c, gpu_rf, gpu_rb, gpu_rklow, tconv,
                             item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratx2_kernel(gpu_c, gpu_rf, gpu_rb, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                ratx4_kernel(gpu_c, gpu_rf, gpu_rb, item_ct1);
             });
      });

      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                qssa_kernel(gpu_rf, gpu_rb, gpu_a, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range = sycl::range<3>(blks) * sycl::range<3>(thrds);
         auto dpct_local_range = sycl::range<3>(thrds);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                qssab_kernel(gpu_rf, gpu_rb, gpu_a, item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                qssa2_kernel(gpu_rf, gpu_rb, gpu_a, item_ct1);
             });
      });

      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                              item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot2_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                               item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot3_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                               item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot6_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                               item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot7_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                               item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot8_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                               item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot9_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                               item_ct1);
             });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         auto dpct_global_range =
             sycl::range<3>(blks2) * sycl::range<3>(thrds2);
         auto dpct_local_range = sycl::range<3>(thrds2);

         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                              dpct_global_range.get(1),
                                              dpct_global_range.get(0)),
                               sycl::range<3>(dpct_local_range.get(2),
                                              dpct_local_range.get(1),
                                              dpct_local_range.get(0))),
             [=](sycl::nd_item<3> item_ct1) {
                rdwdot10_kernel(gpu_rf, gpu_rb, gpu_wdot, rateconv, gpu_molwt,
                                item_ct1);
             });
      });
    }
    // Copy back result
   /*
   DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (q_ct1.memcpy(host_wdot, gpu_wdot, WDOT_SIZE * n * sizeof(real)).wait(),
        0));

    // Free GPU memory
   /*
   DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_t, q_ct1), 0));
   /*
   DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_p, q_ct1), 0));
   /*
   DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_y, q_ct1), 0));
   /*
   DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_wdot, q_ct1), 0));
   /*
   DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_rf, q_ct1), 0));
   /*
   DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_rb, q_ct1), 0));
   /*
   DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_c, q_ct1), 0));
   /*
   DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_rklow, q_ct1), 0));
   /*
   DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_a, q_ct1), 0));
   /*
   DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_eg, q_ct1), 0));
   /*
   DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(gpu_molwt, q_ct1), 0));

    for (int i=0; i<WDOT_SIZE; i++) {
        printf("% 23.16E ", host_wdot[i*n]);
        if (i % 3 == 2)
            printf("\n");
    }
    printf("\n");


    // Free host memory
   /*
   DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(host_t, q_ct1), 0));
   /*
   DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(host_p, q_ct1), 0));
   /*
   DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(host_y, q_ct1), 0));
   /*
   DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(host_wdot, q_ct1), 0));
   /*
   DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((sycl::free(host_molwt, q_ct1), 0));
}
catch (sycl::exception const &exc) {
   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
             << ", line:" << __LINE__ << std::endl;
   std::exit(1);
}
