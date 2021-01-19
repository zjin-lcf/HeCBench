#define DPCT_USM_LEVEL_NONE
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
    auto batch_count = cont.size() / PE_NUM;

    control_dt *h_control;
    anchor_dt *h_arg;
    return_dt *h_ret;

    h_control = (control_dt *)malloc(cont.size() * sizeof(control_dt));
    h_arg = (anchor_dt *)malloc(arg.size() * sizeof(anchor_dt));
    h_ret = (return_dt *)malloc(batch_count * TILE_SIZE * PE_NUM *
                                sizeof(return_dt));
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

    d_control =
        (control_dt *)dpct::dpct_malloc(cont.size() * sizeof(control_dt));
    d_arg = (anchor_dt *)dpct::dpct_malloc(arg.size() * sizeof(anchor_dt));
    d_ret = (return_dt *)dpct::dpct_malloc(batch_count * TILE_SIZE * PE_NUM *
                                           sizeof(return_dt));

    for (auto pe = 0; pe < PE_NUM; pe++) {
        d_max_tracker[pe] = (score_dt *)dpct::dpct_malloc(
            BACK_SEARCH_COUNT_GPU * sizeof(score_dt));
        d_j_tracker[pe] = (parent_dt *)dpct::dpct_malloc(BACK_SEARCH_COUNT_GPU *
                                                         sizeof(parent_dt));
    }
    d_d_max_tracker =
        (score_dt **)dpct::dpct_malloc(PE_NUM * sizeof(score_dt *));
    d_d_j_tracker =
        (parent_dt **)dpct::dpct_malloc(PE_NUM * sizeof(parent_dt *));

    dpct::dpct_memcpy(d_control, h_control, cont.size() * sizeof(control_dt),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_arg, h_arg, arg.size() * sizeof(anchor_dt),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_d_max_tracker, d_max_tracker,
                      PE_NUM * sizeof(score_dt *), dpct::host_to_device);
    dpct::dpct_memcpy(d_d_j_tracker, d_j_tracker, PE_NUM * sizeof(parent_dt *),
                      dpct::host_to_device);

    for (auto batch = 0; batch < batch_count; batch++) {
            std::pair<dpct::buffer_t, size_t>
                d_ret_batch_PE_NUM_TILE_SIZE_buf_ct0 =
                    dpct::get_buffer_and_offset(d_ret +
                                                batch * PE_NUM * TILE_SIZE);
            size_t d_ret_batch_PE_NUM_TILE_SIZE_offset_ct0 =
                d_ret_batch_PE_NUM_TILE_SIZE_buf_ct0.second;
            std::pair<dpct::buffer_t, size_t>
                d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_buf_ct1 =
                    dpct::get_buffer_and_offset(d_arg + batch * PE_NUM *
                                                            TILE_SIZE_ACTUAL);
            size_t d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_offset_ct1 =
                d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_buf_ct1.second;
            std::pair<dpct::buffer_t, size_t> d_control_batch_PE_NUM_buf_ct2 =
                dpct::get_buffer_and_offset(d_control + batch * PE_NUM);
            size_t d_control_batch_PE_NUM_offset_ct2 =
                d_control_batch_PE_NUM_buf_ct2.second;
            dpct::buffer_t d_d_max_tracker_buf_ct3 =
                dpct::get_buffer(d_d_max_tracker);
            dpct::buffer_t d_d_j_tracker_buf_ct4 =
                dpct::get_buffer(d_d_j_tracker);
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
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
                  auto d_ret_batch_PE_NUM_TILE_SIZE_acc_ct0 =
                      d_ret_batch_PE_NUM_TILE_SIZE_buf_ct0.first
                          .get_access<sycl::access::mode::read_write>(cgh);
                  auto d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_acc_ct1 =
                      d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_buf_ct1.first
                          .get_access<sycl::access::mode::read_write>(cgh);
                  auto d_control_batch_PE_NUM_acc_ct2 =
                      d_control_batch_PE_NUM_buf_ct2.first
                          .get_access<sycl::access::mode::read_write>(cgh);
                  auto d_d_max_tracker_acc_ct3 =
                      d_d_max_tracker_buf_ct3
                          .get_access<sycl::access::mode::read_write>(cgh);
                  auto d_d_j_tracker_acc_ct4 =
                      d_d_j_tracker_buf_ct4
                          .get_access<sycl::access::mode::read_write>(cgh);

                  cgh.parallel_for(
                      sycl::nd_range<3>(
                          sycl::range<3>(1, 1, BLOCK_NUM) *
                              sycl::range<3>(1, 1, BACK_SEARCH_COUNT_GPU),
                          sycl::range<3>(1, 1, BACK_SEARCH_COUNT_GPU)),
                      [=](sycl::nd_item<3> item_ct1) {
                            return_dt *d_ret_batch_PE_NUM_TILE_SIZE_ct0 =
                                (return_dt
                                     *)(&d_ret_batch_PE_NUM_TILE_SIZE_acc_ct0
                                            [0] +
                                        d_ret_batch_PE_NUM_TILE_SIZE_offset_ct0);
                            const anchor_dt *d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_ct1 =
                                (const anchor_dt
                                     *)(&d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_acc_ct1
                                            [0] +
                                        d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_offset_ct1);
                            const control_dt *d_control_batch_PE_NUM_ct2 =
                                (const control_dt
                                     *)(&d_control_batch_PE_NUM_acc_ct2[0] +
                                        d_control_batch_PE_NUM_offset_ct2);
                            device_chain_tiled(
                                d_ret_batch_PE_NUM_TILE_SIZE_ct0,
                                d_arg_batch_PE_NUM_TILE_SIZE_ACTUAL_ct1,
                                d_control_batch_PE_NUM_ct2,
                                (score_dt **)(&d_d_max_tracker_acc_ct3[0]),
                                (parent_dt **)(&d_d_j_tracker_acc_ct4[0]),
                                max_dist_x, max_dist_y, bw, item_ct1,
                                active_acc_ct1.get_pointer(),
                                max_tracker_acc_ct1.get_pointer(),
                                j_tracker_acc_ct1.get_pointer());
                      });
            });
    }

    dpct::dpct_memcpy(h_ret, d_ret,
                      batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt),
                      dpct::device_to_host);

    dpct::dpct_free(d_control);
    dpct::dpct_free(d_arg);
    dpct::dpct_free(d_ret);
    for (auto pe = 0; pe < PE_NUM; pe++) {
        dpct::dpct_free(d_max_tracker[pe]);
        dpct::dpct_free(d_j_tracker[pe]);
    }
    dpct::dpct_free(d_d_max_tracker);
    dpct::dpct_free(d_d_j_tracker);

    clock_gettime(CLOCK_BOOTTIME, &end);
    printf(" ***** offloading took %f seconds for end-to-end\n",
        ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / 1E9);

    memcpy(ret.data(), h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));
}

