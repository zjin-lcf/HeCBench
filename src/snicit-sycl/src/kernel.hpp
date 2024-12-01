#include <sycl/sycl.hpp>
#include "atomicAdd.hpp"
#pragma once

namespace SNICIT_BEY{

void y_star_gen(
    const float* Y0,
    int *y_star_row,
    const int num_input,
    const int neurons,
    const int seed_size,
    const sycl::nd_item<3> &item,
    uint8_t *dpct_local) {
    int row_idx =
        item.get_local_id(1) * num_input / item.get_local_range(1);
    int tid = item.get_local_id(2) +
              item.get_local_id(1) * item.get_local_range(2);
    auto shRow = (float *)dpct_local; // combined diff_arr and tmp_star_row here
    if (item.get_local_id(2) == 0) {
        shRow[neurons + seed_size + item.get_local_id(1)] = (float)row_idx;
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < seed_size; i++) {
        if (shRow[neurons+seed_size+i]!=-1.0) {
            if (tid < neurons) {
                shRow[tid] = Y0[neurons*(int)shRow[neurons+seed_size+i]+tid]; // to be compared
            }
            if (tid < seed_size) {
                shRow[neurons+tid] = 0;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (shRow[neurons + seed_size + item.get_local_id(1)] != -1.f) {
                for (int j = item.get_local_id(2); j < neurons;
                     j += item.get_local_range(2)) {
                    if (sycl::fabs(Y0[neurons * row_idx + j] - shRow[j]) > 0.03f) {
                        atomicAdd(shRow[neurons + item.get_local_id(1)], 1.f);
                    }
                }
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (item.get_local_id(1) != i &&
                shRow[neurons + item.get_local_id(1)] < neurons * 0.03f) {
                shRow[neurons + seed_size + item.get_local_id(1)] = -1.f;
            }
            item.barrier(sycl::access::fence_space::local_space);
        }
    }
    if (tid < seed_size) {
        y_star_row[tid] = (int)shRow[neurons+seed_size+tid];
    }
    item.barrier(sycl::access::fence_space::local_space);
}

void coarse_cluster(
    float* Y0,
    const int *y_star_row,
    bool *ne_record,
    const int y_star_cnt,
    int *centroid_LUT,
    const int neurons,
    const sycl::nd_item<3> &item,
    uint8_t *dpct_local) {
    if (centroid_LUT[item.get_group(2)] == -1) {
        ne_record[item.get_group(2)] = true;
        return;
    }
    auto thisRow = (float *)dpct_local;
    // __shared__ float diff_arr[60]; // estimated max y* num
    int tid = item.get_local_id(2) +
              item.get_local_id(1) * item.get_local_range(2);
    if (tid < neurons) {
        thisRow[tid] = Y0[item.get_group(2) * neurons + tid];
    }
    if (tid < y_star_cnt) {
        thisRow[neurons+tid] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int i = item.get_local_id(2); i < neurons;
         i += item.get_local_range(2)) {
        if (sycl::fabs(Y0[neurons * y_star_row[item.get_local_id(1)] + i] -
                       thisRow[i]) > 0.04f) {
            atomicAdd(thisRow[neurons + item.get_local_id(1)], 1.f);
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    int argmin=-10;
    float min_num = neurons+1;
    if (tid == 0) {
        for (int i = 0; i < y_star_cnt; i++) {
            if (min_num > thisRow[neurons+i]) {
                min_num = thisRow[neurons+i];
                argmin = y_star_row[i];
            }
        }
        centroid_LUT[item.get_group(2)] = argmin;
    }
    item.barrier(sycl::access::fence_space::local_space);
    argmin = centroid_LUT[item.get_group(2)];
    float v = ((tid < neurons) &&
               (sycl::fabs(thisRow[tid] - Y0[neurons * argmin + tid]) > 0.04f))
                  ? thisRow[tid] - Y0[neurons * argmin + tid]
                  : 0;
    if (tid < neurons) {
        Y0[item.get_group(2) * neurons + tid] = v; // change blockIdx.x to argmin
    }
    int count =
        (item.barrier(sycl::access::fence_space::local_space),
         sycl::reduce_over_group(item.get_group(), v > 0 ? 1 : 0,
                                 sycl::ext::oneapi::plus<>()));
    if (tid == 0) {
        if (count == 0) ne_record[item.get_group(2)] = false;
        else ne_record[item.get_group(2)] = true;
    }

}

void sparse_hidden_post(
    const int *rowsY,
    const float* Y0,
    const int* roffW,
    const int* colsW,
    const float* valsW,
    const int M, const int N, const int K,
    float* Y1,
    const sycl::nd_item<3> &item,
    uint8_t *dpct_local) {
    auto shRow = (float *)dpct_local;
    int tid = item.get_local_id(2) +
              item.get_local_id(1) * item.get_local_range(2);
    int rid = rowsY[item.get_group(2)];
    if (tid < K) {
        shRow[tid] = 0; 
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int i = item.get_local_id(1); i < N;
         i += item.get_local_range(1)) {
        float valY = Y0[rid * N + i];
        if(valY == 0) {
            continue;
        }

        int begOffW = roffW[i] + item.get_local_id(2);
        int endOffW = roffW[i + 1];
        for (int k = begOffW; k < endOffW; k += item.get_local_range(2)) { // += blockDim.x
            int colW = colsW[k];
            float valW = valsW[k];
            atomicAdd(shRow[colW], valY * valW);
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (tid < K) {
        Y1[rid * K+tid] = shRow[tid];
    }
}

void update_post(
    const int *rowsY,
    const int *centroid_LUT,
    const float* Y0,
    const float* bias,
    const int neurons,
    bool* ne_record,
    float* Y1,
    const sycl::nd_item<3> &item)
{
    int tid = item.get_local_id(2);
    int rid = rowsY[item.get_group(2)];
    float b = bias[item.get_local_id(2)];
    if (centroid_LUT[rid] == -1) {
        Y1[rid * neurons + tid] = sycl::min(
            float(1.0), sycl::max(float(0), Y0[rid * neurons + tid] + b)); // Y0[rid * neurons+tid];
        ne_record[rid] = true;
        return;
    }
    float wy_centroid = Y0[neurons * centroid_LUT[rid] + tid];
    float wdelta_y = Y0[neurons * rid + tid];
    float true_diff =
        sycl::min(float(1.0), sycl::max(float(0), wy_centroid + b + wdelta_y)) -
        sycl::min(float(1.0), sycl::max(float(0), wy_centroid + b));
    float val = (sycl::fabs(true_diff) > 0.05f) ? true_diff : 0;
    int count =
        (item.barrier(sycl::access::fence_space::local_space),
         sycl::reduce_over_group(item.get_group(), val != 0 ? 1 : 0,
                                 sycl::ext::oneapi::plus<>()));
    Y1[rid * neurons+tid] = val;
    if (tid == 0) {
        if (count == 0) ne_record[rid] = false;
        else ne_record[rid] = true;
    }
}

void recover(
    float* Y0,
    const int *centroid_LUT,
    const int neurons,
    const sycl::nd_item<3> &item,
    uint8_t *dpct_local) {
    auto shRow = (float *)dpct_local;
    if (centroid_LUT[item.get_group(2)] == -1) {
        return;
    }
    int tid = item.get_local_id(2);
    shRow[tid] = Y0[item.get_group(2) * neurons + tid] +
                 Y0[centroid_LUT[item.get_group(2)] * neurons + tid];
    item.barrier(sycl::access::fence_space::local_space);
    Y0[item.get_group(2) * neurons + tid] = shRow[tid];
}

}
