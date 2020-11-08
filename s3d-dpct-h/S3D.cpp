#define DPCT_USM_LEVEL_NONE
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
    int probSizes_SP[4] = { 24, 32, 40, 48};
    int probSizes_DP[4] = { 16, 24, 32, 40};
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
   CUDA_SAFE_CALL((host_t = (real *)malloc(n * sizeof(real)), 0));
   /*
   DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((host_p = (real *)malloc(n * sizeof(real)), 0));
   /*
   DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((host_y = (real *)malloc(Y_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (host_wdot = (real *)malloc(WDOT_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((host_molwt = (real *)malloc(WDOT_SIZE * sizeof(real)), 0));

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
   CUDA_SAFE_CALL((dpct::dpct_malloc((void **)&gpu_t, n * sizeof(real)), 0));
   /*
   DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_malloc((void **)&gpu_p, n * sizeof(real)), 0));
   /*
   DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_y, Y_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((
       dpct::dpct_malloc((void **)&gpu_wdot, WDOT_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_rf, RF_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_rb, RB_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_rklow, RKLOW_SIZE * n * sizeof(real)),
        0));
   /*
   DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_c, C_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_a, A_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_eg, EG_SIZE * n * sizeof(real)), 0));
   /*
   DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_malloc((void **)&gpu_molwt, WDOT_SIZE * sizeof(real)), 0));

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
   CUDA_SAFE_CALL((dpct::async_dpct_memcpy(gpu_t, host_t, n * sizeof(real),
                                           dpct::host_to_device),
                   0));
   /*
   DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::async_dpct_memcpy(gpu_p, host_p, n * sizeof(real),
                                           dpct::host_to_device),
                   0));
   /*
   DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::async_dpct_memcpy(gpu_y, host_y, Y_SIZE * n * sizeof(real),
                                dpct::host_to_device),
        0));
   /*
   DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::async_dpct_memcpy(gpu_molwt, host_molwt, WDOT_SIZE * sizeof(real),
                                dpct::host_to_device),
        0));

    unsigned int passes = op.getOptionInt("passes");
    for (unsigned int i = 0; i < passes; i++)
    {
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   ratt_kernel(gpu_t_ct0, gpu_rf_ct1, tconv, item_ct1);
                });
         });
      }

      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct1 = gpu_eg_buf_ct1.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct1 =
                gpu_eg_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_eg_ct1 =
                       (real *)(&gpu_eg_acc_ct1[0] + gpu_eg_offset_ct1);
                   rdsmh_kernel(gpu_t_ct0, gpu_eg_ct1, tconv, item_ct1);
                });
         });
      }

      {
         std::pair<dpct::buffer_t, size_t> gpu_p_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_p);
         size_t gpu_p_offset_ct0 = gpu_p_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct1 = gpu_t_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_y_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_y);
         size_t gpu_y_offset_ct2 = gpu_y_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_c_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_c);
         size_t gpu_c_offset_ct3 = gpu_c_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_p_acc_ct0 =
                gpu_p_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_t_acc_ct1 =
                gpu_t_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_y_acc_ct2 =
                gpu_y_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_c_acc_ct3 =
                gpu_c_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_p_ct0 =
                       (real *)(&gpu_p_acc_ct0[0] + gpu_p_offset_ct0);
                   real *gpu_t_ct1 =
                       (real *)(&gpu_t_acc_ct1[0] + gpu_t_offset_ct1);
                   real *gpu_y_ct2 =
                       (real *)(&gpu_y_acc_ct2[0] + gpu_y_offset_ct2);
                   real *gpu_c_ct3 =
                       (real *)(&gpu_c_acc_ct3[0] + gpu_c_offset_ct3);
                   gr_base(gpu_p_ct0, gpu_t_ct1, gpu_y_ct2, gpu_c_ct3, tconv,
                           pconv, item_ct1);
                });
         });
      }

      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt2_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt3_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt4_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt5_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt6_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt7_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt8_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_eg_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_eg);
         size_t gpu_eg_offset_ct3 = gpu_eg_buf_ct3.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_eg_acc_ct3 =
                gpu_eg_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   real *gpu_eg_ct3 =
                       (real *)(&gpu_eg_acc_ct3[0] + gpu_eg_offset_ct3);
                   ratt9_kernel(gpu_t_ct0, gpu_rf_ct1, gpu_rb_ct2, gpu_eg_ct3,
                                tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rklow_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rklow);
         size_t gpu_rklow_offset_ct1 = gpu_rklow_buf_ct1.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rklow_acc_ct1 =
                gpu_rklow_buf_ct1.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_rklow_ct1 =
                       (real *)(&gpu_rklow_acc_ct1[0] + gpu_rklow_offset_ct1);
                   ratt10_kernel(gpu_t_ct0, gpu_rklow_ct1, tconv, item_ct1);
                });
         });
      }

      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_c_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_c);
         size_t gpu_c_offset_ct1 = gpu_c_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct2 = gpu_rf_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct3 = gpu_rb_buf_ct3.second;
         std::pair<dpct::buffer_t, size_t> gpu_rklow_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_rklow);
         size_t gpu_rklow_offset_ct4 = gpu_rklow_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_c_acc_ct1 =
                gpu_c_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct2 =
                gpu_rf_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct3 =
                gpu_rb_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rklow_acc_ct4 =
                gpu_rklow_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

            auto dpct_global_range =
                sycl::range<3>(blks) * sycl::range<3>(thrds);
            auto dpct_local_range = sycl::range<3>(thrds);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dpct_local_range.get(2),
                                                 dpct_local_range.get(1),
                                                 dpct_local_range.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_c_ct1 =
                       (real *)(&gpu_c_acc_ct1[0] + gpu_c_offset_ct1);
                   real *gpu_rf_ct2 =
                       (real *)(&gpu_rf_acc_ct2[0] + gpu_rf_offset_ct2);
                   real *gpu_rb_ct3 =
                       (real *)(&gpu_rb_acc_ct3[0] + gpu_rb_offset_ct3);
                   real *gpu_rklow_ct4 =
                       (real *)(&gpu_rklow_acc_ct4[0] + gpu_rklow_offset_ct4);
                   ratx_kernel(gpu_t_ct0, gpu_c_ct1, gpu_rf_ct2, gpu_rb_ct3,
                               gpu_rklow_ct4, tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_t_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_t);
         size_t gpu_t_offset_ct0 = gpu_t_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_c_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_c);
         size_t gpu_c_offset_ct1 = gpu_c_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct2 = gpu_rf_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct3 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct3 = gpu_rb_buf_ct3.second;
         std::pair<dpct::buffer_t, size_t> gpu_rklow_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_rklow);
         size_t gpu_rklow_offset_ct4 = gpu_rklow_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_t_acc_ct0 =
                gpu_t_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_c_acc_ct1 =
                gpu_c_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct2 =
                gpu_rf_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct3 =
                gpu_rb_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rklow_acc_ct4 =
                gpu_rklow_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

            auto dpct_global_range =
                sycl::range<3>(blks) * sycl::range<3>(thrds);
            auto dpct_local_range = sycl::range<3>(thrds);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dpct_local_range.get(2),
                                                 dpct_local_range.get(1),
                                                 dpct_local_range.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                   real *gpu_t_ct0 =
                       (real *)(&gpu_t_acc_ct0[0] + gpu_t_offset_ct0);
                   real *gpu_c_ct1 =
                       (real *)(&gpu_c_acc_ct1[0] + gpu_c_offset_ct1);
                   real *gpu_rf_ct2 =
                       (real *)(&gpu_rf_acc_ct2[0] + gpu_rf_offset_ct2);
                   real *gpu_rb_ct3 =
                       (real *)(&gpu_rb_acc_ct3[0] + gpu_rb_offset_ct3);
                   real *gpu_rklow_ct4 =
                       (real *)(&gpu_rklow_acc_ct4[0] + gpu_rklow_offset_ct4);
                   ratxb_kernel(gpu_t_ct0, gpu_c_ct1, gpu_rf_ct2, gpu_rb_ct3,
                                gpu_rklow_ct4, tconv, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_c_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_c);
         size_t gpu_c_offset_ct0 = gpu_c_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_c_acc_ct0 =
                gpu_c_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_c_ct0 =
                       (real *)(&gpu_c_acc_ct0[0] + gpu_c_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   ratx2_kernel(gpu_c_ct0, gpu_rf_ct1, gpu_rb_ct2, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_c_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_c);
         size_t gpu_c_offset_ct0 = gpu_c_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct1 = gpu_rf_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct2 = gpu_rb_buf_ct2.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_c_acc_ct0 =
                gpu_c_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rf_acc_ct1 =
                gpu_rf_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct2 =
                gpu_rb_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_c_ct0 =
                       (real *)(&gpu_c_acc_ct0[0] + gpu_c_offset_ct0);
                   real *gpu_rf_ct1 =
                       (real *)(&gpu_rf_acc_ct1[0] + gpu_rf_offset_ct1);
                   real *gpu_rb_ct2 =
                       (real *)(&gpu_rb_acc_ct2[0] + gpu_rb_offset_ct2);
                   ratx4_kernel(gpu_c_ct0, gpu_rf_ct1, gpu_rb_ct2, item_ct1);
                });
         });
      }

      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_a_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_a);
         size_t gpu_a_offset_ct2 = gpu_a_buf_ct2.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_a_acc_ct2 =
                gpu_a_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_a_ct2 =
                       (real *)(&gpu_a_acc_ct2[0] + gpu_a_offset_ct2);
                   qssa_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_a_ct2, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_a_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_a);
         size_t gpu_a_offset_ct2 = gpu_a_buf_ct2.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_a_acc_ct2 =
                gpu_a_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);

            auto dpct_global_range =
                sycl::range<3>(blks) * sycl::range<3>(thrds);
            auto dpct_local_range = sycl::range<3>(thrds);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dpct_local_range.get(2),
                                                 dpct_local_range.get(1),
                                                 dpct_local_range.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_a_ct2 =
                       (real *)(&gpu_a_acc_ct2[0] + gpu_a_offset_ct2);
                   qssab_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_a_ct2, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_a_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_a);
         size_t gpu_a_offset_ct2 = gpu_a_buf_ct2.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_a_acc_ct2 =
                gpu_a_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_a_ct2 =
                       (real *)(&gpu_a_acc_ct2[0] + gpu_a_offset_ct2);
                   qssa2_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_a_ct2, item_ct1);
                });
         });
      }

      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2, rateconv,
                                 gpu_molwt_ct4, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot2_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2,
                                  rateconv, gpu_molwt_ct4, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot3_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2,
                                  rateconv, gpu_molwt_ct4, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot6_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2,
                                  rateconv, gpu_molwt_ct4, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot7_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2,
                                  rateconv, gpu_molwt_ct4, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot8_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2,
                                  rateconv, gpu_molwt_ct4, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot9_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2,
                                  rateconv, gpu_molwt_ct4, item_ct1);
                });
         });
      }
      {
         std::pair<dpct::buffer_t, size_t> gpu_rf_buf_ct0 =
             dpct::get_buffer_and_offset(gpu_rf);
         size_t gpu_rf_offset_ct0 = gpu_rf_buf_ct0.second;
         std::pair<dpct::buffer_t, size_t> gpu_rb_buf_ct1 =
             dpct::get_buffer_and_offset(gpu_rb);
         size_t gpu_rb_offset_ct1 = gpu_rb_buf_ct1.second;
         std::pair<dpct::buffer_t, size_t> gpu_wdot_buf_ct2 =
             dpct::get_buffer_and_offset(gpu_wdot);
         size_t gpu_wdot_offset_ct2 = gpu_wdot_buf_ct2.second;
         std::pair<dpct::buffer_t, size_t> gpu_molwt_buf_ct4 =
             dpct::get_buffer_and_offset(gpu_molwt);
         size_t gpu_molwt_offset_ct4 = gpu_molwt_buf_ct4.second;
         q_ct1.submit([&](sycl::handler &cgh) {
            auto gpu_rf_acc_ct0 =
                gpu_rf_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_rb_acc_ct1 =
                gpu_rb_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto gpu_wdot_acc_ct2 =
                gpu_wdot_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto gpu_molwt_acc_ct4 =
                gpu_molwt_buf_ct4.first
                    .get_access<sycl::access::mode::read_write>(cgh);

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
                   real *gpu_rf_ct0 =
                       (real *)(&gpu_rf_acc_ct0[0] + gpu_rf_offset_ct0);
                   real *gpu_rb_ct1 =
                       (real *)(&gpu_rb_acc_ct1[0] + gpu_rb_offset_ct1);
                   real *gpu_wdot_ct2 =
                       (real *)(&gpu_wdot_acc_ct2[0] + gpu_wdot_offset_ct2);
                   real *gpu_molwt_ct4 =
                       (real *)(&gpu_molwt_acc_ct4[0] + gpu_molwt_offset_ct4);
                   rdwdot10_kernel(gpu_rf_ct0, gpu_rb_ct1, gpu_wdot_ct2,
                                   rateconv, gpu_molwt_ct4, item_ct1);
                });
         });
      }
    }
    // Copy back result
   /*
   DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL(
       (dpct::dpct_memcpy(host_wdot, gpu_wdot, WDOT_SIZE * n * sizeof(real),
                          dpct::device_to_host),
        0));

    // Free GPU memory
   /*
   DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_t), 0));
   /*
   DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_p), 0));
   /*
   DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_y), 0));
   /*
   DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_wdot), 0));
   /*
   DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_rf), 0));
   /*
   DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_rb), 0));
   /*
   DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_c), 0));
   /*
   DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_rklow), 0));
   /*
   DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_a), 0));
   /*
   DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_eg), 0));
   /*
   DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((dpct::dpct_free(gpu_molwt), 0));

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
   CUDA_SAFE_CALL((free(host_t), 0));
   /*
   DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((free(host_p), 0));
   /*
   DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((free(host_y), 0));
   /*
   DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((free(host_wdot), 0));
   /*
   DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted. You
   may need to rewrite this code.
   */
   CUDA_SAFE_CALL((free(host_molwt), 0));
}
catch (sycl::exception const &exc) {
   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
             << ", line:" << __LINE__ << std::endl;
   std::exit(1);
}
