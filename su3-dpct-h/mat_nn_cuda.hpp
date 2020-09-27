// Cuda implementation
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

/*
DPCT1009:1: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define CUCHECK(err, s)

#define THREADS_PER_SITE 36

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
void k_mat_nn(
  const site*       __restrict__ a,
  const su3_matrix* __restrict__ b,
        site*       __restrict__ c,
  int               total_sites,
  sycl::nd_item<3> item_ct1)
{
  int id = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
           item_ct1.get_local_id(2);
  int i = id/36;

  if (i < total_sites) {
    int j = (id%36)/9;
    int k = (id%9)/3;
    int l = id%3;
    Complx cc = {0.0, 0.0};
    for (int m=0;m<3;m++)
#ifdef MILC_COMPLEX
      CMULSUM(a[i].link[j].e[k][m], b[j].e[m][l], cc);
#else
      cc += a[i].link[j].e[k][m] * b[j].e[m][l];
#endif
    c[i].link[j].e[k][l] = cc;

  }
}


#ifdef MILC_COMPLEX
double su3_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b,
                  std::vector<site> &c,
#else
double su3_mat_nn(thrust::host_vector<site> &a,
                  thrust::host_vector<su3_matrix> &b,
                  thrust::host_vector<site> &c, 
#endif
                  size_t total_sites, int iterations, int threadsPerBlock,
                  int use_device) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  int blocksPerGrid;
  int size_a = sizeof(site) * total_sites;
  int size_b = sizeof(su3_matrix) * 4;
  int size_c = sizeof(site) * total_sites;

  if (threadsPerBlock == 0)
    threadsPerBlock = THREADS_PER_SITE;

  // Declare target storage and copy A and B
  int cuErr;
  site *d_a, *d_c;
  su3_matrix *d_b;
  /*
  DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cuErr = (dpct::dpct_malloc((void **)&d_a, size_a), 0);
  CUCHECK(cuErr, "Unable to allocate array d_a");
  /*
  DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cuErr = (dpct::dpct_malloc((void **)&d_b, size_b), 0);
  CUCHECK(cuErr, "Unable to allocate array d_b");
  /*
  DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cuErr = (dpct::dpct_malloc((void **)&d_c, size_c), 0);
  CUCHECK(cuErr, "Unable to allocate array d_c");
  dpct::dpct_memcpy(d_a, a.data(), size_a, dpct::host_to_device);
  dpct::dpct_memcpy(d_b, b.data(), size_b, dpct::host_to_device);

  blocksPerGrid = total_sites;

  if (verbose >= 1) {
    printf("Number of blocks set to %d\n", blocksPerGrid);
    printf("Threads per block set to %d\n", threadsPerBlock);
  }

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      dev_ct1.queues_wait_and_throw();
      tstart = Clock::now();
	  }
    {
      dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
      dpct::buffer_t d_b_buf_ct1 = dpct::get_buffer(d_b);
      dpct::buffer_t d_c_buf_ct2 = dpct::get_buffer(d_c);
      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_a_acc_ct0 =
            d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
        auto d_b_acc_ct1 =
            d_b_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
        auto d_c_acc_ct2 =
            d_c_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
                                  sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
              k_mat_nn((const site *)(&d_a_acc_ct0[0]),
                       (const dsu3_matrix *)(&d_b_acc_ct1[0]),
                       (site *)(&d_c_acc_ct2[0]), total_sites, item_ct1);
            });
      });
    }
  }
  dev_ct1.queues_wait_and_throw();
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  /*
  DPCT1010:4: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUCHECK(0, "k_mat_nn kernel Failed");

  // copy data back from device
  dpct::dpct_memcpy(c.data(), d_c, size_c, dpct::device_to_host);

  // Deallocate
  dpct::dpct_free(d_a);
  dpct::dpct_free(d_b);
  dpct::dpct_free(d_c);

  return (ttotal /= 1.0e6);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
