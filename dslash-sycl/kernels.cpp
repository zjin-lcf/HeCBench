#include "dslash.h"
#include "common.h"

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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  
  // Set the loop and work-group parameters
  size_t total_sites = sites_on_node; 
  size_t total_even_sites = even_sites_on_node;
  
  auto copy_start = Clock::now();

  // allocate device memory
  buffer<su3_vector, 1> d_src (src.data(), total_sites * 1);
  buffer<su3_matrix, 1> d_fat (fat.data(), total_sites * 4);
  buffer<su3_matrix, 1> d_lng (lng.data(), total_sites * 4);
  buffer<su3_vector, 1> d_dst (total_sites * 1);
  buffer<su3_matrix, 1> d_fatbck (total_sites * 4);
  buffer<su3_matrix, 1> d_lngbck (total_sites * 4);

  // allocate offsets for device gathers and copy to shared buffers
  buffer<size_t, 1> d_fwd (fwd, total_sites * 4);
  buffer<size_t, 1> d_bck (bck, total_sites * 4);
  buffer<size_t, 1> d_fwd3 (fwd3, total_sites * 4);
  buffer<size_t, 1> d_bck3 (bck3, total_sites * 4);
  
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
  q.submit( [&](handler& cgh) {
    auto d_fat_acc = d_fat.get_access<sycl_read>(cgh);
    auto d_bck_acc = d_bck.get_access<sycl_read>(cgh);
    auto d_lng_acc = d_lng.get_access<sycl_read>(cgh);
    auto d_bck3_acc = d_bck3.get_access<sycl_read>(cgh);
    auto d_fatbck_acc = d_fatbck.get_access<sycl_discard_write>(cgh);
    auto d_lngbck_acc = d_lngbck.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class make_back>(nd_range<1> {total_wi, wgsize}, [=](nd_item<1> item) {
      size_t mySite = item.get_global_id(0);
      if (mySite < total_even_sites) {
	for(int dir = 0; dir < 4; dir++) {
	  su3_adjoint( d_fat_acc.get_pointer() + 4*d_bck_acc[4*mySite+dir]+dir, 
                       d_fatbck_acc.get_pointer() + 4*mySite+dir );
	  su3_adjoint( d_lng_acc.get_pointer() + 4*d_bck3_acc[4*mySite+dir]+dir, 
                       d_lngbck_acc.get_pointer() + 4*mySite+dir );
	}
      }
    }); // end of the kernel lambda function
  });   // end of command group
  q.wait();
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
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      q.wait();
      tstart = Clock::now();
    } 
    // Dslash kernel
    q.submit( [&](handler& cgh) {
      auto d_fat_acc = d_fat.get_access<sycl_read>(cgh);
      auto d_src_acc = d_src.get_access<sycl_read>(cgh);
      auto d_fwd_acc = d_fwd.get_access<sycl_read>(cgh);
      auto d_fwd3_acc = d_fwd3.get_access<sycl_read>(cgh);
      auto d_bck3_acc = d_bck3.get_access<sycl_read>(cgh);
      auto d_bck_acc = d_bck.get_access<sycl_read>(cgh);
      auto d_lng_acc = d_lng.get_access<sycl_read>(cgh);
      auto d_lngbck_acc = d_lngbck.get_access<sycl_read>(cgh);
      auto d_fatbck_acc = d_fatbck.get_access<sycl_read>(cgh);
      auto d_dst_acc = d_dst.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class dslash>(nd_range<1> {total_wi, wgsize}, [=](nd_item<1> item) {
	size_t mySite = item.get_global_id(0);
	if (mySite < total_even_sites) {
	  su3_vector v;
          for (size_t k=0; k<4; ++k) {
            auto a = d_fat_acc.get_pointer() + mySite*4 + k;
	    auto b = d_src_acc.get_pointer() + d_fwd_acc[4*mySite + k];
            if (k == 0)
              mult_su3_mat_vec(a, b, &d_dst_acc[mySite]);
            else 
              mult_su3_mat_vec_sum(a, b, &d_dst_acc[mySite]);
          }
          for (size_t k=0; k<4; ++k) {
            auto a = d_lng_acc.get_pointer() + mySite*4 + k;
	    auto b = d_src_acc.get_pointer() + d_fwd3_acc[4*mySite + k];
            if (k == 0) 
              mult_su3_mat_vec(a, b, &v);
            else
              mult_su3_mat_vec_sum(a, b, &v);
          }
	  add_su3_vector(&d_dst_acc[mySite], &v, &d_dst_acc[mySite]);
          for (size_t k=0; k<4; ++k) {
            auto a = d_fatbck_acc.get_pointer() + mySite*4 + k;
	    auto b = d_src_acc.get_pointer() + d_bck_acc[4*mySite + k];
            if (k == 0) 
              mult_su3_mat_vec(a, b, &v);
            else
              mult_su3_mat_vec_sum(a, b, &v);
          }
	  sub_su3_vector(&d_dst_acc[mySite], &v, &d_dst_acc[mySite]);
          for (size_t k=0; k<4; ++k) {
            auto a = d_lngbck_acc.get_pointer() + mySite*4 + k;
            auto b = d_src_acc.get_pointer() + d_bck3_acc[4*mySite + k];
            if (k == 0) 
              mult_su3_mat_vec(a, b, &v);
            else
              mult_su3_mat_vec_sum(a, b, &v);
          }
	  sub_su3_vector(&d_dst_acc[mySite], &v, &d_dst_acc[mySite]);
	} // end of if mySite
      }); // end of the kernel lambda function
    });   // end of command group
    q.wait();
  } // end of iteration loop
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(
                  Clock::now()-tstart).count();
    
  // Move the result back to the host
  copy_start = Clock::now();
    
  q.submit( [&](handler& cgh) {
    auto acc = d_dst.get_access<sycl_read>(cgh);
    cgh.copy(acc, dst.data());
  });
  q.submit( [&](handler& cgh) {
    auto acc = d_fatbck.get_access<sycl_read>(cgh);
    cgh.copy(acc, fatbck.data());
  });
  q.submit( [&](handler& cgh) {
    auto acc = d_lngbck.get_access<sycl_read>(cgh);
    cgh.copy(acc, lngbck.data());
  }); 
  q.wait();

  copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
  if (verbose > 1) {
    std::cout << "Time to offload backward links = " << copy_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  return (ttotal /= 1.0e6);
}
