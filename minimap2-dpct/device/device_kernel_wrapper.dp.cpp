#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <string>
#include <ctime>
#include <cstdio>
#include "device_kernel_wrapper.h"
#include "datatypes.h"
#include "kernel_common.h"
#include "memory_scheduler.h"

SYCL_EXTERNAL
void device_chain_tiled(
        return_dt *ret, const anchor_dt *a,
        const control_dt *control, score_dt **max_tracker_g, parent_dt **j_tracker_g,
        const int max_dist_x, const int max_dist_y, const int bw,
        sycl::nd_item<3> item_ct1, anchor_dt *active, score_dt *max_tracker,
        parent_dt *j_tracker);


void device_chain_kernel_wrapper(
        std::vector<control_dt> &cont,
        std::vector<anchor_dt> &arg,
        std::vector<return_dt> &ret,
        int max_dist_x, int max_dist_y, int bw)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    auto batch_count = cont.size() / PE_NUM;

    control_dt *h_control;
    anchor_dt *h_arg;
    return_dt *h_ret;

    h_control = sycl::malloc_host<control_dt>(cont.size(), q_ct1);
    h_arg = sycl::malloc_host<anchor_dt>(arg.size(), q_ct1);
    h_ret =
        sycl::malloc_host<return_dt>(batch_count * TILE_SIZE * PE_NUM, q_ct1);
    ret.resize(batch_count * TILE_SIZE * PE_NUM);

    memcpy(h_control, cont.data(), cont.size() * sizeof(control_dt));
    memcpy(h_arg, arg.data(), arg.size() * sizeof(anchor_dt));

    struct timespec start, end;
    clock_gettime(CLOCK_BOOTTIME, &start);

    control_dt *d_control;
    anchor_dt *d_arg;
    return_dt *d_ret;

    // presistent storage
    score_dt *d_max_tracker[PE_NUM];
    parent_dt *d_j_tracker[PE_NUM];

    score_dt **d_d_max_tracker;
    parent_dt **d_d_j_tracker;

    d_control = sycl::malloc_device<control_dt>(cont.size(), q_ct1);
    d_arg = sycl::malloc_device<anchor_dt>(arg.size(), q_ct1);
    d_ret =
        sycl::malloc_device<return_dt>(batch_count * TILE_SIZE * PE_NUM, q_ct1);

    for (auto pe = 0; pe < PE_NUM; pe++) {
        d_max_tracker[pe] =
            sycl::malloc_device<score_dt>(BACK_SEARCH_COUNT_GPU, q_ct1);
        d_j_tracker[pe] =
            sycl::malloc_device<parent_dt>(BACK_SEARCH_COUNT_GPU, q_ct1);
    }
    d_d_max_tracker = sycl::malloc_device<score_dt *>(PE_NUM, q_ct1);
    d_d_j_tracker = sycl::malloc_device<parent_dt *>(PE_NUM, q_ct1);

    q_ct1.memcpy(d_control, h_control, cont.size() * sizeof(control_dt)).wait();
    q_ct1.memcpy(d_arg, h_arg, arg.size() * sizeof(anchor_dt)).wait();
    q_ct1.memcpy(d_d_max_tracker, d_max_tracker, PE_NUM * sizeof(score_dt *))
        .wait();
    q_ct1.memcpy(d_d_j_tracker, d_j_tracker, PE_NUM * sizeof(parent_dt *))
        .wait();

    for (auto batch = 0; batch < batch_count; batch++) {
            q_ct1.submit([&](sycl::handler &cgh) {
                  sycl::accessor<anchor_dt, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      active_acc_ct1(
                          sycl::range<1>(64 /*BACK_SEARCH_COUNT_GPU*/), cgh);
                  sycl::accessor<score_dt, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      max_tracker_acc_ct1(
                          sycl::range<1>(64 /*BACK_SEARCH_COUNT_GPU*/), cgh);
                  sycl::accessor<parent_dt, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      j_tracker_acc_ct1(
                          sycl::range<1>(64 /*BACK_SEARCH_COUNT_GPU*/), cgh);

                  cgh.parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, BLOCK_NUM) *
                              sycl::range<3>(1, 1, BACK_SEARCH_COUNT_GPU),
                          sycl::range<3>(1, 1, BACK_SEARCH_COUNT_GPU)),
                      [=](sycl::nd_item<3> item_ct1) {
                            device_chain_tiled(
                                d_ret + batch * PE_NUM * TILE_SIZE,
                                d_arg + batch * PE_NUM * TILE_SIZE_ACTUAL,
                                d_control + batch * PE_NUM, d_d_max_tracker,
                                d_d_j_tracker, max_dist_x, max_dist_y, bw,
                                item_ct1, active_acc_ct1.get_pointer(),
                                max_tracker_acc_ct1.get_pointer(),
                                j_tracker_acc_ct1.get_pointer());
                      });
            });
    }

    q_ct1
        .memcpy(h_ret, d_ret,
                batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt))
        .wait();

    sycl::free(d_control, q_ct1);
    sycl::free(d_arg, q_ct1);
    sycl::free(d_ret, q_ct1);
    for (auto pe = 0; pe < PE_NUM; pe++) {
        sycl::free(d_max_tracker[pe], q_ct1);
        sycl::free(d_j_tracker[pe], q_ct1);
    }
    sycl::free(d_d_max_tracker, q_ct1);
    sycl::free(d_d_j_tracker, q_ct1);

    clock_gettime(CLOCK_BOOTTIME, &end);
    printf(" ***** offloading took %f seconds for end-to-end\n",
        ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / 1E9);

    memcpy(ret.data(), h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));
}

