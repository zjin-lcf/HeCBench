#include <cmath>
#include <sycl/sycl.hpp>
#include "atomicAdd.hpp"
#pragma once

namespace SNICIT_BEY{
void dense_input(
  const float* Y0,
  const float* weight,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1,
  const sycl::nd_item<3> &item, uint8_t *dpct_local) {
    auto shRow = (float *)dpct_local;
    if (item.get_local_id(1) == 0) {
        shRow[item.get_local_id(2)] = bias[item.get_local_id(2)];
    }

    item.barrier(sycl::access::fence_space::local_space);
    for (int i = item.get_local_id(1); i < N; i += item.get_local_range(1)) {
        float valY = Y0[item.get_group(2) * N + i];
        if(valY == 0) {
            continue;
        }
        float valW = weight[i * K + item.get_local_id(2)];
        atomicAdd(shRow[item.get_local_id(2)], valY * valW);
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (item.get_local_id(1) == 0) {
        Y1[item.get_group(2) * K + item.get_local_id(2)] =
            sycl::min(float(1.0), (float)(sycl::max(float(0), (float)(shRow[item.get_local_id(2)]))));
    }
}


void sparse_hidden(
    
  const float* Y0,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float* bias,
  const int M, const int N, const int K, float* Y1,
  const sycl::nd_item<3> &item, uint8_t *dpct_local)
{
    // (8, 128)
    auto shRow = (float *)dpct_local;
    int tid = item.get_local_id(2) +
              item.get_local_id(1) * item.get_local_range(2);
    if (tid < K) {
        shRow[tid] = bias[tid]; 
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int i = item.get_local_id(1); i < N;
         i += item.get_local_range(1)) {
        float valY = Y0[item.get_group(2) * N + i];
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
        Y1[item.get_group(2) * K + tid] =
            sycl::min(float(1.0), sycl::max(float(0), shRow[tid]));
    }
}

void dense_output(
  const float* Y0,
  const float* weight,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1,
  const sycl::nd_item<3> &item, uint8_t *dpct_local)
{
    auto shRow = (float *)dpct_local;
    if (item.get_local_id(1) == 0) {
        shRow[item.get_local_id(2)] = bias[item.get_local_id(2)];
    }

    item.barrier(sycl::access::fence_space::local_space);
    for (int i = item.get_local_id(1); i < N;
         i += item.get_local_range(1)) {
        float valY = Y0[item.get_group(2) * N + i];
        if(valY == 0) {
            continue;
        }
        float valW = weight[i * K + item.get_local_id(2)];
        atomicAdd(shRow[item.get_local_id(2)], valY * valW);
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (item.get_local_id(1) == 0) {
        Y1[item.get_group(2) * K + item.get_local_id(2)] =
            shRow[item.get_local_id(2)];
    }
}

void check_acc(
    float* Y, int num_classes, int num_input, int* label, int *cnt,
    const sycl::nd_item<3> &item, uint8_t *dpct_local) {
    auto shcnt = (int *)dpct_local;
    if (item.get_local_id(2) == 0)
        shcnt[0] = 0;
    item.barrier(sycl::access::fence_space::local_space);
    for (int i = item.get_local_id(2); i < num_input;
         i += item.get_local_range(2)) {
        int argmax = 0;
        float tmpmax = -10000.0;
        for (int j = 0; j < num_classes; j++) {
            if (Y[i*num_classes+j] > tmpmax) {
                argmax = j;
                tmpmax = Y[i*num_classes+j];
            }
        }
        // printf("label[i]=%d argmax=%d, tmpmax = %f\n", label[i], argmax, tmpmax);
        if (argmax == label[i])
            atomicAdd(shcnt[0], 1);
    }
    // printf("shcnt[0]=%d", shcnt[0]);
    item.barrier(sycl::access::fence_space::local_space);
    if (item.get_local_id(2) == 0) 
       cnt[0] = shcnt[0];
}

}
