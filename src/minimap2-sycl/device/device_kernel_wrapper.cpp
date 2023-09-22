#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "datatypes.h"
#include "kernel_common.h"
#include "memory_scheduler.h"
#include <sycl/sycl.hpp>
#include "device_kernel.cpp"

void device_chain_kernel_wrapper(
    std::vector<control_dt> &cont,
    std::vector<anchor_dt> &arg,
    std::vector<return_dt> &ret,
    int max_dist_x, int max_dist_y, int bw)
{
  auto batch_count = cont.size() / PE_NUM;

  control_dt *h_control = (control_dt*) malloc (sizeof(control_dt) * cont.size());
  anchor_dt *h_arg = (anchor_dt*) malloc (arg.size() * sizeof(anchor_dt));
  return_dt *h_ret = (return_dt*) malloc (batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

  ret.resize(batch_count * TILE_SIZE * PE_NUM);

  memcpy(h_control, cont.data(), cont.size() * sizeof(control_dt));
  memcpy(h_arg, arg.data(), arg.size() * sizeof(anchor_dt));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  score_dt *d_max_tracker = sycl::malloc_device<score_dt>(PE_NUM * BACK_SEARCH_COUNT_GPU, q);
  parent_dt *d_j_tracker = sycl::malloc_device<parent_dt>(PE_NUM * BACK_SEARCH_COUNT_GPU, q);

  const int control_size = cont.size();
  const int arg_size = arg.size();

  control_dt *d_control = sycl::malloc_device<control_dt>(control_size, q);
  anchor_dt *d_arg = sycl::malloc_device<anchor_dt>(arg_size, q);
  return_dt *d_ret = sycl::malloc_device<return_dt>(batch_count * TILE_SIZE * PE_NUM, q);

  q.memcpy(d_control, h_control, control_size * sizeof(control_dt));
  q.memcpy(d_arg, h_arg, arg_size * sizeof(anchor_dt));

  sycl::range<1> gws (BLOCK_NUM * THREAD_FACTOR * BACK_SEARCH_COUNT_GPU);
  sycl::range<1> lws (THREAD_FACTOR * BACK_SEARCH_COUNT_GPU);

  q.wait();
  auto k_start = std::chrono::steady_clock::now();

  for (auto batch = 0; batch < batch_count; batch++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<anchor_dt, 1> active_sm(sycl::range<1>(BACK_SEARCH_COUNT_GPU), cgh);
      sycl::local_accessor< score_dt, 1> max_tracker_sm(sycl::range<1>(BACK_SEARCH_COUNT_GPU), cgh);
      sycl::local_accessor<parent_dt, 1> j_tracker_sm(sycl::range<1>(BACK_SEARCH_COUNT_GPU), cgh);

      cgh.parallel_for<class tiled_chain>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        device_chain_tiled(d_ret + batch * PE_NUM * TILE_SIZE,
                           d_arg + batch * PE_NUM * TILE_SIZE_ACTUAL,
                           d_control + batch * PE_NUM,
                           d_max_tracker,
                           d_j_tracker,
                           active_sm.get_pointer(),
                           max_tracker_sm.get_pointer(),
                           j_tracker_sm.get_pointer(),
                           item,
                           max_dist_x, max_dist_y, bw);

      });
    });
  }

  q.wait();
  auto k_end = std::chrono::steady_clock::now();
  auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("Total kernel execution time: %f (s)\n", k_time * 1e-9);

  q.memcpy(h_ret, d_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

  sycl::free(d_control, q);
  sycl::free(d_arg, q);
  sycl::free(d_ret, q);
  sycl::free(d_max_tracker, q);
  sycl::free(d_j_tracker, q);

  memcpy(ret.data(), h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

  free(h_control);
  free(h_arg);
  free(h_ret);
}
