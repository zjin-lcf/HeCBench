#include "dslash.h"
#include <sycl/sycl.hpp>

// SYCL lambda function kernel names
class make_back;
class dslash;

double dslash_fn(
  const std::vector<su3_vector> &src, 
        std::vector<su3_vector> &dst,
  const std::vector<su3_matrix> &fat,
  const std::vector<su3_matrix> &lng,
        std::vector<su3_matrix> &fatbck,
        std::vector<su3_matrix> &lngbck,
  size_t *fwd, size_t *bck, size_t *fwd3, size_t *bck3,    
  const size_t iterations,
  size_t wgsize )
{ 
  // Set device and queue
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  // Set the loop and work-group parameters
  size_t total_sites = sites_on_node; 
  size_t total_even_sites = even_sites_on_node;
  
  auto copy_start = Clock::now();

  // allocate device memory
  su3_vector *d_src = sycl::malloc_device<su3_vector>(total_sites * 1, q);
  q.memcpy(d_src, src.data(), total_sites * 1 * sizeof(su3_vector));

  su3_matrix *d_fat  = sycl::malloc_device<su3_matrix>(total_sites * 4, q);
  q.memcpy(d_fat, fat.data(), total_sites * 4 * sizeof(su3_matrix));

  su3_matrix *d_lng  = sycl::malloc_device<su3_matrix>(total_sites * 4, q);
  q.memcpy(d_lng, lng.data(), total_sites * 4 * sizeof(su3_matrix));

  su3_vector *d_dst  = sycl::malloc_device<su3_vector>(total_sites * 1, q);
  su3_matrix *d_fatbck  = sycl::malloc_device<su3_matrix>(total_sites * 4, q);
  su3_matrix *d_lngbck  = sycl::malloc_device<su3_matrix>(total_sites * 4, q);

  // allocate offsets for device gathers and copy to shared buffers
  size_t *d_fwd  = sycl::malloc_device<size_t>(total_sites * 4, q);
  q.memcpy(d_fwd, fwd, total_sites * 4 * sizeof(size_t));

  size_t *d_bck  = sycl::malloc_device<size_t>(total_sites * 4, q);
  q.memcpy(d_bck, bck, total_sites * 4 * sizeof(size_t));

  size_t *d_fwd3  = sycl::malloc_device<size_t>(total_sites * 4, q);
  q.memcpy(d_fwd3, fwd3, total_sites * 4 * sizeof(size_t));

  size_t *d_bck3  = sycl::malloc_device<size_t>(total_sites * 4, q);
  q.memcpy(d_bck3, bck3, total_sites * 4 * sizeof(size_t));

  q.wait();
  
  double copy_time = std::chrono::duration_cast<std::chrono::microseconds>(
                     Clock::now()-copy_start).count();
  if (verbose > 1) {
    std::cout << "Time to offload input data = " << copy_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  // Create backward links on the device
  size_t total_wi = total_even_sites;
  if (verbose > 1) {
    std::cout << "Creating backward links"  << std::endl;
    std::cout << "Setting number of work items " << total_wi << std::endl;
    std::cout << "Setting workgroup size to " << 1 << std::endl;
  }
  auto back_start = Clock::now();
  q.submit( [&](sycl::handler& cgh) {
    cgh.parallel_for<class make_back>(sycl::nd_range<1> {total_wi, wgsize}, [=](sycl::nd_item<1> item) {
      size_t mySite = item.get_global_id(0);
      if (mySite < total_even_sites) {
	for(int dir = 0; dir < 4; dir++) {
	  su3_adjoint( d_fat + 4*d_bck[4*mySite+dir]+dir, 
                       d_fatbck + 4*mySite+dir );
	  su3_adjoint( d_lng + 4*d_bck3[4*mySite+dir]+dir, 
                       d_lngbck + 4*mySite+dir );
	}
      }
    }); // end of the kernel lambda function
  }).wait();   // end of command group
  double back_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-back_start).count();
  if (verbose > 1) {
    std::cout << "Time to create back links = " << back_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  // Dslash benchmark loop
  total_wi = total_even_sites;
  if (verbose > 0) {
    std::cout << "Running dslash loop" << std::endl;
    std::cout << "Setting number of work items to " << total_wi << std::endl;
    std::cout << "Setting workgroup size to " << wgsize << std::endl;
  }
  auto tstart = Clock::now();
  for (size_t iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      q.wait();
      tstart = Clock::now();
    }
    // Dslash kernel
    q.submit( [&](sycl::handler& cgh) {
      cgh.parallel_for<class dslash>(
        sycl::nd_range<1> {total_wi, wgsize}, [=](sycl::nd_item<1> item) {
	size_t mySite = item.get_global_id(0);
	if (mySite < total_even_sites) {
	  su3_vector v;
          for (size_t k=0; k<4; ++k) {
            auto a = d_fat + mySite*4 + k;
	    auto b = d_src + d_fwd[4*mySite + k];
            if (k == 0)
              mult_su3_mat_vec(a, b, &d_dst[mySite]);
            else 
              mult_su3_mat_vec_sum(a, b, &d_dst[mySite]);
          }
          for (size_t k=0; k<4; ++k) {
            auto a = d_lng + mySite*4 + k;
	    auto b = d_src + d_fwd3[4*mySite + k];
            if (k == 0) 
              mult_su3_mat_vec(a, b, &v);
            else
              mult_su3_mat_vec_sum(a, b, &v);
          }
	  add_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
          for (size_t k=0; k<4; ++k) {
            auto a = d_fatbck + mySite*4 + k;
	    auto b = d_src + d_bck[4*mySite + k];
            if (k == 0) 
              mult_su3_mat_vec(a, b, &v);
            else
              mult_su3_mat_vec_sum(a, b, &v);
          }
	  sub_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
          for (size_t k=0; k<4; ++k) {
            auto a = d_lngbck + mySite*4 + k;
            auto b = d_src + d_bck3[4*mySite + k];
            if (k == 0) 
              mult_su3_mat_vec(a, b, &v);
            else
              mult_su3_mat_vec_sum(a, b, &v);
          }
	  sub_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
	} // end of if mySite
      }); // end of the kernel lambda function
    }).wait();   // end of command group
  } // end of iteration loop
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(
                  Clock::now()-tstart).count();
    
  // Move the result back to the host
  copy_start = Clock::now();
    
  q.memcpy(dst.data(), d_dst, total_sites * 1 * sizeof(su3_vector));
  q.memcpy(fatbck.data(), d_fatbck, total_sites * 4 * sizeof(su3_matrix));
  q.memcpy(lngbck.data(), d_lngbck, total_sites * 4 * sizeof(su3_matrix));
  q.wait();

  copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
  if (verbose > 1) {
    std::cout << "Time to offload backward links = " << copy_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  sycl::free(d_src, q);
  sycl::free(d_fat, q);
  sycl::free(d_lng, q);
  sycl::free(d_dst, q);
  sycl::free(d_fatbck, q);
  sycl::free(d_lngbck, q);
  sycl::free(d_fwd, q);
  sycl::free(d_bck, q);
  sycl::free(d_fwd3, q);
  sycl::free(d_bck3, q);

  return (ttotal /= 1.0e6);
}
