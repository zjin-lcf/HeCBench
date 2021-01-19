#include <vector>
#include <string>
#include <ctime>
#include <cstdio>
#include "device_kernel_wrapper.h"
#include "datatypes.h"
#include "common.h"
#include "memory_scheduler.h"


__global__
void device_chain_tiled(
        return_dt *ret, const anchor_dt *a,
        const control_dt *control, score_dt **max_tracker, parent_dt **j_tracker,
        const int max_dist_x, const int max_dist_y, const int bw);

__host__
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

    cudaMallocHost(&h_control, cont.size() * sizeof(control_dt));
    cudaMallocHost(&h_arg, arg.size() * sizeof(anchor_dt));
    cudaMallocHost(&h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));
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

    cudaMalloc(&d_control, cont.size() * sizeof(control_dt));
    cudaMalloc(&d_arg, arg.size() * sizeof(anchor_dt));
    cudaMalloc(&d_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

    for (auto pe = 0; pe < PE_NUM; pe++) {
        cudaMalloc(&d_max_tracker[pe], BACK_SEARCH_COUNT_GPU * sizeof(score_dt));
        cudaMalloc(&d_j_tracker[pe], BACK_SEARCH_COUNT_GPU * sizeof(parent_dt));
    }
    cudaMalloc(&d_d_max_tracker, PE_NUM * sizeof(score_dt *));
    cudaMalloc(&d_d_j_tracker, PE_NUM * sizeof(parent_dt *));

    cudaMemcpy(d_control, h_control,
            cont.size() * sizeof(control_dt), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arg, h_arg,
            arg.size() * sizeof(anchor_dt), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_max_tracker, d_max_tracker,
            PE_NUM * sizeof(score_dt *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_j_tracker, d_j_tracker,
            PE_NUM * sizeof(parent_dt *), cudaMemcpyHostToDevice);


    for (auto batch = 0; batch < batch_count; batch++) {
            device_chain_tiled<<<BLOCK_NUM, BACK_SEARCH_COUNT_GPU>>> (
                    d_ret + batch * PE_NUM * TILE_SIZE,
                    d_arg + batch * PE_NUM * TILE_SIZE_ACTUAL,
                    d_control + batch * PE_NUM ,
                    d_d_max_tracker,
                    d_d_j_tracker,
                    max_dist_x, max_dist_y, bw);
    }

    cudaMemcpy(h_ret, d_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt), cudaMemcpyDeviceToHost);

    cudaFree(d_control);
    cudaFree(d_arg);
    cudaFree(d_ret);
    for (auto pe = 0; pe < PE_NUM; pe++) {
        cudaFree(d_max_tracker[pe]);
        cudaFree(d_j_tracker[pe]);
    }
    cudaFree(d_d_max_tracker);
    cudaFree(d_d_j_tracker);

    clock_gettime(CLOCK_BOOTTIME, &end);
    printf(" ***** offloading took %f seconds for end-to-end\n",
        ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / 1E9);

    memcpy(ret.data(), h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));
}

