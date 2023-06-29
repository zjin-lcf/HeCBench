#include <sycl/sycl.hpp>
#define THREADS_PER_SITE 36

class k_mat_nn;

double su3_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c,
                  const size_t total_sites, const size_t iterations, size_t wgsize, const int target)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int size_a = sizeof(site) * total_sites;
  int size_b = sizeof(su3_matrix) * 4;
  int size_c = sizeof(site) * total_sites;

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

  site *d_a = sycl::malloc_device<site>(total_sites, q);
  q.memcpy(d_a, a.data(), size_a);

  su3_matrix *d_b = sycl::malloc_device<su3_matrix>(4, q);
  q.memcpy(d_b, b.data(), size_b);

  site *d_c = sycl::malloc_device<site>(total_sites, q);

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      q.wait();
      tstart = Clock::now();
    }

    // create a command_group to issue commands
    q.submit([&](sycl::handler& cgh) {
      // Lambda function defines the kernel scope
      cgh.parallel_for<class k_mat_nn>(
        sycl::nd_range<1> {total_wi, wgsize}, [=](sycl::nd_item<1> item) {
        size_t id = item.get_global_id(0);
        size_t i = id/36;
        if (i < total_sites) {
          int j = (id%36)/9;
          int k = (id%9)/3;
          int l = id%3;
          Complx cc = {0.0, 0.0};
          for (int m=0;m<3;m++) {
            #ifdef MILC_COMPLEX
            CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
            #else
            cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
            #endif
          }
          d_c[i].link[j].e[k][l] = cc;
        }
      }); // end of the kernel lambda function
    });   // end of command group
  } // end of iteration loop
  q.wait();

  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  // copy data back from device
  q.memcpy(c.data(), d_c, size_c).wait();

  // Deallocate
  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);

  return (ttotal /= 1.0e6);
} // end of SYCL block
