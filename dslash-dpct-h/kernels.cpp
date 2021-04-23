#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dslash.h"

void make_back(
    const su3_matrix* d_fat,
    const su3_matrix* d_lng,
    const size_t* d_bck, 
    const size_t* d_bck3,
          su3_matrix* d_fatbck,
          su3_matrix* d_lngbck,
    const int total_even_sites,
    sycl::nd_item<3> item_ct1)
{
  size_t mySite = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  if (mySite < total_even_sites) {
    for(int dir = 0; dir < 4; dir++) {
      su3_adjoint( d_fat + 4*d_bck[4*mySite+dir]+dir, 
          d_fatbck + 4*mySite+dir );
      su3_adjoint( d_lng + 4*d_bck3[4*mySite+dir]+dir, 
          d_lngbck + 4*mySite+dir );
    }
  }
}

void dslash (
    const su3_matrix* d_fat,
    const su3_matrix* d_lng,
    const su3_matrix* d_fatbck,
    const su3_matrix* d_lngbck,
    const su3_vector* d_src,
          su3_vector* d_dst,
    const size_t* d_fwd,
    const size_t* d_bck,
    const size_t* d_fwd3,
    const size_t* d_bck3,
    const int total_even_sites,
    sycl::nd_item<3> item_ct1)
{
  size_t mySite = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
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
}

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // Set the loop and work-group parameters
  size_t total_sites = sites_on_node; 
  size_t total_even_sites = even_sites_on_node;

  auto copy_start = Clock::now();

  // allocate device memory
  su3_vector* d_src;
  d_src =
      (dsu3_vector *)dpct::dpct_malloc(total_sites * 1 * sizeof(su3_vector));
  dpct::dpct_memcpy(d_src, src.data(), total_sites * 1 * sizeof(su3_vector),
                    dpct::host_to_device);

  su3_matrix* d_fat;
  d_fat =
      (dsu3_matrix *)dpct::dpct_malloc(total_sites * 4 * sizeof(su3_matrix));
  dpct::dpct_memcpy(d_fat, fat.data(), total_sites * 4 * sizeof(su3_matrix),
                    dpct::host_to_device);

  su3_matrix* d_lng;
  d_lng =
      (dsu3_matrix *)dpct::dpct_malloc(total_sites * 4 * sizeof(su3_matrix));
  dpct::dpct_memcpy(d_lng, lng.data(), total_sites * 4 * sizeof(su3_matrix),
                    dpct::host_to_device);

  su3_vector* d_dst;
  d_dst =
      (dsu3_vector *)dpct::dpct_malloc(total_sites * 1 * sizeof(su3_vector));

  su3_matrix* d_fatbck;
  d_fatbck =
      (dsu3_matrix *)dpct::dpct_malloc(total_sites * 4 * sizeof(su3_matrix));

  su3_matrix* d_lngbck;
  d_lngbck =
      (dsu3_matrix *)dpct::dpct_malloc(total_sites * 4 * sizeof(su3_matrix));

  size_t* d_fwd;
  d_fwd = (size_t *)dpct::dpct_malloc(total_sites * 4 * sizeof(size_t));
  dpct::dpct_memcpy(d_fwd, fwd, total_sites * 4 * sizeof(size_t),
                    dpct::host_to_device);

  size_t* d_bck;
  d_bck = (size_t *)dpct::dpct_malloc(total_sites * 4 * sizeof(size_t));
  dpct::dpct_memcpy(d_bck, bck, total_sites * 4 * sizeof(size_t),
                    dpct::host_to_device);

  size_t* d_fwd3;
  d_fwd3 = (size_t *)dpct::dpct_malloc(total_sites * 4 * sizeof(size_t));
  dpct::dpct_memcpy(d_fwd3, fwd3, total_sites * 4 * sizeof(size_t),
                    dpct::host_to_device);

  size_t* d_bck3;
  d_bck3 = (size_t *)dpct::dpct_malloc(total_sites * 4 * sizeof(size_t));
  dpct::dpct_memcpy(d_bck3, bck3, total_sites * 4 * sizeof(size_t),
                    dpct::host_to_device);

  // allocate offsets for device gathers and copy to shared buffers

  double copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
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

  {
    dpct::buffer_t d_fat_buf_ct0 = dpct::get_buffer(d_fat);
    dpct::buffer_t d_lng_buf_ct1 = dpct::get_buffer(d_lng);
    dpct::buffer_t d_bck_buf_ct2 = dpct::get_buffer(d_bck);
    dpct::buffer_t d_bck3_buf_ct3 = dpct::get_buffer(d_bck3);
    dpct::buffer_t d_fatbck_buf_ct4 = dpct::get_buffer(d_fatbck);
    dpct::buffer_t d_lngbck_buf_ct5 = dpct::get_buffer(d_lngbck);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_fat_acc_ct0 =
          d_fat_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto d_lng_acc_ct1 =
          d_lng_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto d_bck_acc_ct2 =
          d_bck_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
      auto d_bck3_acc_ct3 =
          d_bck3_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
      auto d_fatbck_acc_ct4 =
          d_fatbck_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
      auto d_lngbck_acc_ct5 =
          d_lngbck_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, total_wi),
                                         sycl::range<3>(1, 1, 1)),
                       [=](sycl::nd_item<3> item_ct1) {
                         make_back((const dsu3_matrix *)(&d_fat_acc_ct0[0]),
                                   (const dsu3_matrix *)(&d_lng_acc_ct1[0]),
                                   (const size_t *)(&d_bck_acc_ct2[0]),
                                   (const size_t *)(&d_bck3_acc_ct3[0]),
                                   (dsu3_matrix *)(&d_fatbck_acc_ct4[0]),
                                   (dsu3_matrix *)(&d_lngbck_acc_ct5[0]),
                                   total_even_sites, item_ct1);
                       });
    });
  }

  dev_ct1.queues_wait_and_throw();
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
      dev_ct1.queues_wait_and_throw();
      tstart = Clock::now();
    } 
    // Dslash kernel
    sycl::range<3> grid(1, 1, (total_wi + wgsize - 1) / wgsize);
    sycl::range<3> block(1, 1, wgsize);
    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    {
      dpct::buffer_t d_fat_buf_ct0 = dpct::get_buffer(d_fat);
      dpct::buffer_t d_lng_buf_ct1 = dpct::get_buffer(d_lng);
      dpct::buffer_t d_fatbck_buf_ct2 = dpct::get_buffer(d_fatbck);
      dpct::buffer_t d_lngbck_buf_ct3 = dpct::get_buffer(d_lngbck);
      dpct::buffer_t d_src_buf_ct4 = dpct::get_buffer(d_src);
      dpct::buffer_t d_dst_buf_ct5 = dpct::get_buffer(d_dst);
      dpct::buffer_t d_fwd_buf_ct6 = dpct::get_buffer(d_fwd);
      dpct::buffer_t d_bck_buf_ct7 = dpct::get_buffer(d_bck);
      dpct::buffer_t d_fwd3_buf_ct8 = dpct::get_buffer(d_fwd3);
      dpct::buffer_t d_bck3_buf_ct9 = dpct::get_buffer(d_bck3);
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_fat_acc_ct0 =
            d_fat_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
        auto d_lng_acc_ct1 =
            d_lng_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
        auto d_fatbck_acc_ct2 =
            d_fatbck_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
        auto d_lngbck_acc_ct3 =
            d_lngbck_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
        auto d_src_acc_ct4 =
            d_src_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
        auto d_dst_acc_ct5 =
            d_dst_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
        auto d_fwd_acc_ct6 =
            d_fwd_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
        auto d_bck_acc_ct7 =
            d_bck_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);
        auto d_fwd3_acc_ct8 =
            d_fwd3_buf_ct8.get_access<sycl::access::mode::read_write>(cgh);
        auto d_bck3_acc_ct9 =
            d_bck3_buf_ct9.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                           dslash((const dsu3_matrix *)(&d_fat_acc_ct0[0]),
                                  (const dsu3_matrix *)(&d_lng_acc_ct1[0]),
                                  (const dsu3_matrix *)(&d_fatbck_acc_ct2[0]),
                                  (const dsu3_matrix *)(&d_lngbck_acc_ct3[0]),
                                  (const dsu3_vector *)(&d_src_acc_ct4[0]),
                                  (dsu3_vector *)(&d_dst_acc_ct5[0]),
                                  (const size_t *)(&d_fwd_acc_ct6[0]),
                                  (const size_t *)(&d_bck_acc_ct7[0]),
                                  (const size_t *)(&d_fwd3_acc_ct8[0]),
                                  (const size_t *)(&d_bck3_acc_ct9[0]),
                                  total_even_sites, item_ct1);
                         });
      });
    }

    dev_ct1.queues_wait_and_throw();
  } // end of iteration loop
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(
      Clock::now()-tstart).count();

  // Move the result back to the host
  copy_start = Clock::now();

  dpct::dpct_memcpy(dst.data(), d_dst, total_sites * 1 * sizeof(su3_vector),
                    dpct::device_to_host);
  dpct::dpct_memcpy(fatbck.data(), d_fatbck,
                    total_sites * 4 * sizeof(su3_matrix), dpct::device_to_host);
  dpct::dpct_memcpy(lngbck.data(), d_lngbck,
                    total_sites * 4 * sizeof(su3_matrix), dpct::device_to_host);

  copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
  if (verbose > 1) {
    std::cout << "Time to offload backward links = " << copy_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  dpct::dpct_free(d_src);
  dpct::dpct_free(d_fat);
  dpct::dpct_free(d_lng);
  dpct::dpct_free(d_dst);
  dpct::dpct_free(d_fatbck);
  dpct::dpct_free(d_lngbck);
  dpct::dpct_free(d_fwd);
  dpct::dpct_free(d_bck);
  dpct::dpct_free(d_fwd3);
  dpct::dpct_free(d_bck3);

  return (ttotal /= 1.0e6);
}
