#include <sycl/sycl.hpp>

using uchar4 = sycl::uchar4;

/* ################# wrappers ################### */

void compute_costs(
    sycl::queue &q,
    int current_w, int w, int h,
    uchar4 *d_pixels,
    short *d_costs_left,
    short *d_costs_up,
    short *d_costs_right)
{
#ifndef COMPUTE_COSTS_FULL

  sycl::range<2> lws (COSTS_BLOCKSIZE_Y, COSTS_BLOCKSIZE_X);
  sycl::range<2> gws (COSTS_BLOCKSIZE_Y * ((h-1)/(COSTS_BLOCKSIZE_Y-1) + 1),
                      COSTS_BLOCKSIZE_X * ((current_w-1)/(COSTS_BLOCKSIZE_X-2) + 1));

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<pixel, 1> sm (sycl::range<1>(COSTS_BLOCKSIZE_Y * COSTS_BLOCKSIZE_X), cgh);
    cgh.parallel_for<class compute_costs>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      compute_costs_kernel(item,
                           sm.get_pointer(),
                           d_pixels,
                           d_costs_left,
                           d_costs_up,
                           d_costs_right,
                           w, h, current_w);
    });
  });

#else

  sycl::range<2> lws (COSTS_BLOCKSIZE_Y, COSTS_BLOCKSIZE_X);
  sycl::range<2> gws (COSTS_BLOCKSIZE_Y * ((h-1)/COSTS_BLOCKSIZE_Y + 1),
                      COSTS_BLOCKSIZE_X * ((current_w-1)/COSTS_BLOCKSIZE_X + 1));

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<pixel, 1> sm (
      sycl::range<1>((COSTS_BLOCKSIZE_Y+1) * (COSTS_BLOCKSIZE_X+2)), cgh);
    cgh.parallel_for<class compute_costs_full>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      compute_costs_full_kernel(item,
                                sm.get_pointer(),
                                d_pixels,
                                d_costs_left,
                                d_costs_up,
                                d_costs_right,
                                w, h, current_w);
    });
  });

#endif
}

void compute_M(
    sycl::queue &q,
    int current_w, int w, int h,
    int *d_M,
    short *d_costs_left,
    short *d_costs_up,
    short *d_costs_right)
{
#if !defined(COMPUTE_M_SINGLE) && !defined(COMPUTE_M_ITERATE)

  if(current_w <= 256){
    sycl::range<1> gws (current_w);
    sycl::range<1> lws (current_w);
    //compute_M_kernel_small<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w);
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int> sm (sycl::range<1>(2*current_w), cgh);
      cgh.parallel_for<class compute_M_small>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        compute_M_kernel_small(item,
                               sm.get_pointer(),
                               d_costs_left,
                               d_costs_up,
                               d_costs_right,
                               d_M,
                               w, h, current_w);
      });
    });
  }
  else{
    sycl::range<1> lws (COMPUTE_M_BLOCKSIZE_X);
    sycl::range<1> gws (COMPUTE_M_BLOCKSIZE_X * ((current_w-1)/COMPUTE_M_BLOCKSIZE_X + 1));
    sycl::range<1> gws2 (COMPUTE_M_BLOCKSIZE_X * ((current_w-COMPUTE_M_BLOCKSIZE_X-1)/COMPUTE_M_BLOCKSIZE_X + 1));

    int num_iterations = (h-1)/(COMPUTE_M_BLOCKSIZE_X/2 - 1) + 1;

    int base_row = 0;
    for(int i = 0; i < num_iterations; i++){
      //compute_M_kernel_step1<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, base_row);
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<int, 1> sm (
          sycl::range<1>(2*COMPUTE_M_BLOCKSIZE_X), cgh);
        cgh.parallel_for<class compute_M_step1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          compute_M_kernel_step1(item,
                                 sm.get_pointer(),
                                 d_costs_left,
                                 d_costs_up,
                                 d_costs_right,
                                 d_M,
                                 w, h, current_w, base_row);
        });
      });

      //compute_M_kernel_step2<<<num_blocks2, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, base_row);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class compute_M_step2>(sycl::nd_range<1>(gws2, lws), [=] (sycl::nd_item<1> item) {
          compute_M_kernel_step2(item,
                                 d_costs_left,
                                 d_costs_up,
                                 d_costs_right,
                                 d_M,
                                 w, h, current_w, base_row);
        });
      });

      base_row = base_row + (COMPUTE_M_BLOCKSIZE_X/2) - 1;
    }
  }

#endif
#ifdef COMPUTE_M_SINGLE

  int block_size = std::min(256, next_pow2(current_w));
  sycl::range<1> lws (block_size);
  sycl::range<1> gws (block_size);

  int num_el = (current_w-1)/block_size + 1;
  //compute_M_kernel_single<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, num_el);
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<int> sm (2*current_w, cgh);
    cgh.parallel_for<class compute_M_single>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      compute_M_kernel_single(item,
                              sm.get_pointer(),
                              d_costs_left,
                              d_costs_up,
                              d_costs_right,
                              d_M,
                              w, h, current_w, num_el);
      });
    });

#else
#ifdef COMPUTE_M_ITERATE

  sycl::range<1> gws (COMPUTE_M_BLOCKSIZE_X * ((current_w-1)/COMPUTE_M_BLOCKSIZE_X + 1));
  sycl::range<1> lws (COMPUTE_M_BLOCKSIZE_X);

  //compute_M_kernel_iterate0<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, current_w);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class compute_M_iter0>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      compute_M_kernel_iterate0(item,
                                d_costs_left,
                                d_costs_up,
                                d_costs_right,
                                d_M,
                                w, current_w);
      });
    });

  for(int row = 1; row < h; row++){
  //  compute_M_kernel_iterate1<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, current_w, row);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class compute_M_iter1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        compute_M_kernel_iterate1(item,
                                  d_costs_left,
                                  d_costs_up,
                                  d_costs_right,
                                  d_M,
                                  w, current_w, row);
        });
      });
    }

#endif
#endif
}

void find_min_index(
    sycl::queue &q,
    int current_w,
    int *d_indices_ref,
    int *d_indices,
    int *reduce_row)
{
  //set the reference index array
  q.memcpy(d_indices, d_indices_ref, current_w*sizeof(int));

  sycl::range<1> lws (REDUCE_BLOCKSIZE_X);
  sycl::range<1> gws (1);

  int reduce_num_elements = current_w;
  do{
    int num_blocks_x = (reduce_num_elements-1)/(REDUCE_BLOCKSIZE_X*REDUCE_ELEMENTS_PER_THREAD) + 1;
    gws[0] = REDUCE_BLOCKSIZE_X * num_blocks_x;
    // min_reduce<<<num_blocks, threads_per_block>>>(reduce_row, d_indices, reduce_num_elements);
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int, 1> sm_val (sycl::range<1>(REDUCE_BLOCKSIZE_X), cgh);
      sycl::local_accessor<int, 1> sm_ix (sycl::range<1>(REDUCE_BLOCKSIZE_X), cgh);
      cgh.parallel_for<class reduce>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        min_reduce(item,
                   sm_val.get_pointer(),
                   sm_ix.get_pointer(),
                   reduce_row,
                   d_indices,
                   reduce_num_elements);
      });
    });
    reduce_num_elements = num_blocks_x;
  }while(reduce_num_elements > 1);
}

void find_seam(
    sycl::queue &q,
    int current_w, int w, int h,
    int *d_M,
    int *d_indices,
    int *d_seam )
{
  //find_seam_kernel<<<1, 1>>>(d_M, d_indices, d_seam, w, h, current_w);
  q.submit([&] (sycl::handler &cgh) {
    cgh.single_task<class find_seam>( [=] () {
      find_seam_kernel(d_M, d_indices, d_seam, w, h, current_w);
    });
  });
}

void remove_seam(
    sycl::queue &q,
    int current_w, int w, int h,
    int *d_M,
    uchar4 *d_pixels,
    uchar4 *d_pixels_swap,
    int *d_seam )
{
  int num_blocks_x = (current_w-1)/REMOVE_BLOCKSIZE_X + 1;
  int num_blocks_y = (h-1)/REMOVE_BLOCKSIZE_Y + 1;
  sycl::range<2> lws (REMOVE_BLOCKSIZE_Y, REMOVE_BLOCKSIZE_X);
  sycl::range<2> gws (REMOVE_BLOCKSIZE_Y * num_blocks_y,
                      REMOVE_BLOCKSIZE_X * num_blocks_x);

  //remove_seam_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_pixels_swap, d_seam, w, h, current_w);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class update_seam>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      remove_seam_kernel(item,
                         d_pixels,
                         d_pixels_swap,
                         d_seam,
                         w, h, current_w);
    });
  });
}

void update_costs(
    sycl::queue &q,
    int current_w, int w, int h,
    int *d_M,
    uchar4 *d_pixels,
    short *d_costs_left,
    short *d_costs_up,
    short *d_costs_right,
    short *d_costs_swap_left,
    short *d_costs_swap_up,
    short *d_costs_swap_right,
    int *d_seam )
{
  int num_blocks_x = (current_w-1)/UPDATE_BLOCKSIZE_X + 1;
  int num_blocks_y = (h-1)/UPDATE_BLOCKSIZE_Y + 1;
  sycl::range<2> lws (UPDATE_BLOCKSIZE_Y, UPDATE_BLOCKSIZE_X);
  sycl::range<2> gws (UPDATE_BLOCKSIZE_Y * num_blocks_y,
                      UPDATE_BLOCKSIZE_X * num_blocks_x);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class update_costs>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      update_costs_kernel (item,
                           d_pixels,
                           d_costs_left,
                           d_costs_up,
                           d_costs_right,
                           d_costs_swap_left,
                           d_costs_swap_up,
                           d_costs_swap_right,
                           d_seam,
                           w, h, current_w);
    });
  });
}

void approx_setup(
    sycl::queue &q,
    int current_w, int w, int h,
    uchar4 *d_pixels,
    int *d_index_map,
    int *d_offset_map,
    int *d_M )
{
  int num_blocks_x = (current_w-1)/(APPROX_SETUP_BLOCKSIZE_X-4) + 1;
  int num_blocks_y = (h-2)/(APPROX_SETUP_BLOCKSIZE_Y-1) + 1;
  sycl::range<2> lws (APPROX_SETUP_BLOCKSIZE_Y, APPROX_SETUP_BLOCKSIZE_X);
  sycl::range<2> gws (num_blocks_y * APPROX_SETUP_BLOCKSIZE_Y,
                      num_blocks_x * APPROX_SETUP_BLOCKSIZE_X);

  sycl::range<1> sm_size (APPROX_SETUP_BLOCKSIZE_Y*APPROX_SETUP_BLOCKSIZE_X);
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<pixel, 1> p_sm (sm_size, cgh);
    sycl::local_accessor<short, 1> l_sm (sm_size, cgh);
    sycl::local_accessor<short, 1> u_sm (sm_size, cgh);
    sycl::local_accessor<short, 1> r_sm (sm_size, cgh);

    cgh.parallel_for<class approx_setup>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      approx_setup_kernel(item,
                          p_sm.get_pointer(),
                          l_sm.get_pointer(),
                          u_sm.get_pointer(),
                          r_sm.get_pointer(),
                          d_pixels,
                          d_index_map,
                          d_offset_map,
                          d_M,
                          w, h, current_w);
    });
  });
}

void approx_M(
    sycl::queue &q,
    int current_w, int w, int h,
    int *d_offset_map,
    int *d_M )
{
  int num_blocks_x = (current_w-1)/APPROX_M_BLOCKSIZE_X + 1;
  int num_blocks_y = h/2;
  sycl::range<2> lws (1, APPROX_M_BLOCKSIZE_X);
  sycl::range<2> gws (h/2, APPROX_M_BLOCKSIZE_X * num_blocks_x);

  int step = 1;
  while(num_blocks_y > 0){
   // approx_M_kernel<<<num_blocks, threads_per_block>>>(d_offset_map, d_M, w, h, current_w, step);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class approx_M>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        approx_M_kernel(item,
                        d_offset_map, d_M,
                        w, h, current_w, step);
      });
    });

    num_blocks_y = num_blocks_y/2;
    step = step*2;
  }
}

void approx_seam(
    sycl::queue &q,
    int w, int h,
    int *d_index_map,
    int *d_indices,
    int *d_seam )
{
  //approx_seam_kernel<<<1, 1>>>(d_index_map, d_indices, d_seam, w, h);
  q.submit([&] (sycl::handler &cgh) {
    cgh.single_task<class approx_seam>( [=] () {
      approx_seam_kernel(d_index_map, d_indices, d_seam, w, h);
    });
  });
}
