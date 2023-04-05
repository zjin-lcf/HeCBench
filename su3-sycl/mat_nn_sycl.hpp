#include "common.h"
#define THREADS_PER_SITE 36

// Sycl requires that kernels be named
class k_mat_nn;

double su3_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c, 
                  const size_t total_sites, const size_t iterations, size_t wgsize, const int target)
{ 
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  // check to make sure the workgroup size is sufficient for the algorithm
  if (wgsize == 0)
    wgsize = THREADS_PER_SITE;

  // set the total number of work items
  size_t total_wi = total_sites * wgsize;
  if (verbose >= 1) {
    std::cout << "Setting number of work items " << total_wi << std::endl;
    std::cout << "Workgroup size is " << wgsize << std::endl;
  }
  std::cout << std::flush;

  // wrap arrays in SYCL buffers, suppling global memory pointer implicitly copies the data to the device when needed
  buffer<site, 1>       a_buf {a.data(), range<1> {total_sites}};
  buffer<su3_matrix, 1> b_buf {b.data(), range<1> {4}};
  buffer<site, 1>       c_buf {range<1> {total_sites}};
  // The copy of c from device -> host will occur when the destructor is called (at the end of the scope)
  c_buf.set_final_data(c.data());

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      q.wait(); 
      tstart = Clock::now();
    }

    // create a command_group to issue commands
    q.submit([&](handler& cgh) {
      // request access to the device buffers
      auto a = a_buf.get_access<sycl_read>(cgh);
      auto b = b_buf.get_access<sycl_read>(cgh);
      auto c = c_buf.get_access<sycl_write>(cgh);

      // Lambda function defines the kernel scope
      cgh.parallel_for<class k_mat_nn>(nd_range<1> {total_wi, wgsize}, [=](nd_item<1> item) {
        size_t id = item.get_global_id(0);
        size_t i = id/36;
        if (i < total_sites) {
          int j = (id%36)/9;
          int k = (id%9)/3;
          int l = id%3;
          Complx cc = {0.0, 0.0};
          for (int m=0;m<3;m++) {
            #ifdef MILC_COMPLEX
            CMULSUM(a[i].link[j].e[k][m], b[j].e[m][l], cc);
            #else
            cc += a[i].link[j].e[k][m] * b[j].e[m][l];
            #endif
          }
          c[i].link[j].e[k][l] = cc;
        }
      }); // end of the kernel lambda function
    });   // end of command group
  } // end of iteration loop
  q.wait();

  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  return (ttotal /= 1.0e6);
} // end of SYCL block

