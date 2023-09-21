#include <omp.h>
#include "dslash.h"

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
  // Set the loop and work-group parameters
  size_t total_sites = sites_on_node; 
  size_t total_even_sites = even_sites_on_node;
  double ttotal; 
  double copy_time;

  std::chrono::time_point<Clock> copy_start = Clock::now();

  // allocate device memory
  const su3_vector* d_src = src.data();
  const su3_matrix* d_fat = fat.data();
  const su3_matrix* d_lng = lng.data();
  su3_vector* d_dst = dst.data(); 
  su3_matrix* d_fatbck = fatbck.data(); 
  su3_matrix* d_lngbck = lngbck.data(); 
  size_t* d_fwd = fwd;
  size_t* d_bck = bck;
  size_t* d_fwd3 = fwd3;
  size_t* d_bck3 = bck3;

  // allocate offsets for device gathers and copy to shared buffers
  //
#pragma omp target data map (to: d_src[0:total_sites*1], \
		                 d_fat[0:total_sites*4], \
		                 d_lng[0:total_sites*4], \
		                 d_fwd[0:total_sites*4], \
		                 d_bck[0:total_sites*4], \
		                 d_fwd3[0:total_sites*4], \
		                 d_bck3[0:total_sites*4]) \
                       map(from: d_dst[0:total_sites],\
                                 d_fatbck[0:total_sites*4],\
                                 d_lngbck[0:total_sites*4])
{

  copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
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

  #pragma omp target teams distribute parallel for thread_limit(1)
  for (size_t mySite = 0; mySite < total_even_sites; mySite++) {
    for(int dir = 0; dir < 4; dir++) {
      su3_adjoint( d_fat + 4*d_bck[4*mySite+dir]+dir, 
          d_fatbck + 4*mySite+dir );
      su3_adjoint( d_lng + 4*d_bck3[4*mySite+dir]+dir, 
          d_lngbck + 4*mySite+dir );
    }
  }

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
      tstart = Clock::now();
    } 
    // Dslash kernel
    #pragma omp target teams distribute parallel for thread_limit(wgsize)
    for (size_t mySite = 0; mySite < total_even_sites; mySite++) {
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
  } // end of iteration loop

  ttotal = std::chrono::duration_cast<std::chrono::microseconds>(
      Clock::now()-tstart).count();

  // Move the result back to the host
  copy_start = Clock::now();
}

  copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
  if (verbose > 1) {
    std::cout << "Time to offload backward links = " << copy_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  return (ttotal /= 1.0e6);
}
