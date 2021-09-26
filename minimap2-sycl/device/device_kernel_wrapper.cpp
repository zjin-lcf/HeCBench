#include <vector>
#include <cstring>
#include <ctime>
#include <cstdio>
#include "datatypes.h"
#include "kernel_common.h"
#include "memory_scheduler.h"
#include "common.h"
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

  struct timespec start, end;
  clock_gettime(CLOCK_BOOTTIME, &start);

  {
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    buffer<score_dt, 1> d_max_tracker (PE_NUM * BACK_SEARCH_COUNT_GPU);
    buffer<parent_dt, 1> d_j_tracker (PE_NUM * BACK_SEARCH_COUNT_GPU);
    d_max_tracker.set_final_data(nullptr);
    d_j_tracker.set_final_data(nullptr);

    int control_size = cont.size();
    int arg_size = arg.size();

    buffer<control_dt, 1> d_control (h_control, cont.size());
    buffer<anchor_dt, 1> d_arg (h_arg, arg.size());
    buffer<return_dt, 1> d_ret (batch_count * TILE_SIZE * PE_NUM);

    range<1> gws (BLOCK_NUM * THREAD_FACTOR * BACK_SEARCH_COUNT_GPU);
    range<1> lws (THREAD_FACTOR * BACK_SEARCH_COUNT_GPU);

    for (auto batch = 0; batch < batch_count; batch++) {
      q.submit([&] (handler &cgh) {
        accessor<anchor_dt, 1, sycl_read_write, access::target::local> active_sm(BACK_SEARCH_COUNT_GPU, cgh);
        accessor< score_dt, 1, sycl_read_write, access::target::local> max_tracker_sm(BACK_SEARCH_COUNT_GPU, cgh);
        accessor<parent_dt, 1, sycl_read_write, access::target::local> j_tracker_sm(BACK_SEARCH_COUNT_GPU, cgh);
        auto ret = d_ret.get_access<sycl_write>(cgh);
        auto arg = d_arg.get_access<sycl_read>(cgh);
        auto cont = d_control.get_access<sycl_read>(cgh);
        auto max_tracker_g = d_max_tracker.get_access<sycl_read_write>(cgh);
        auto j_tracker_g = d_j_tracker.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class tiled_chain>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          device_chain_tiled(ret.get_pointer() + batch * PE_NUM * TILE_SIZE,
                             arg.get_pointer() + batch * PE_NUM * TILE_SIZE_ACTUAL,
                             cont.get_pointer() + batch * PE_NUM,
                             max_tracker_g.get_pointer(),
                             j_tracker_g.get_pointer(),
                             active_sm.get_pointer(),
                             max_tracker_sm.get_pointer(),
                             j_tracker_sm.get_pointer(),
                             item,
                             max_dist_x, max_dist_y, bw);

        });
      });
    }
    q.submit([&] (handler &cgh) {
      auto ret = d_ret.get_access<sycl_read>(cgh);
      cgh.copy(ret, h_ret);
    });
    q.wait();
  }

  clock_gettime(CLOCK_BOOTTIME, &end);
  printf(" ***** offloading took %f seconds for end-to-end\n",
      ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / 1E9);

  memcpy(ret.data(), h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

  free(h_control);
  free(h_arg);
  free(h_ret);
}

