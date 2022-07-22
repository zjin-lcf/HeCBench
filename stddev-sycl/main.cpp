/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"
#include "reference.h"

// final step for the deviation of a sample
template <typename Type, typename IdxType>
class sampleKernel;

// sum of products using atomics
template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
class sopKernel;

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param std the output stddev vector
 * @param data the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 */
template <typename Type, typename IdxType = int>
void stddev(queue &q,
            buffer<Type, 1> &d_std, 
            buffer<const Type, 1> &d_data, 
            IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;

  range<2> gws ((D + (IdxType)ColsPerBlk - 1) / (IdxType)ColsPerBlk ,
                (N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk * TPB);

  range<2> lws (1, TPB);

  // required for atomics
  q.submit([&] (handler &cgh) {
    auto acc = d_std.template get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, (Type)0);
  });

  q.submit([&] (handler &cgh) {
    auto std = d_std.template get_access<sycl_read_write>(cgh);
    auto data = d_data.template get_access<sycl_read>(cgh);
    accessor<Type, 1, sycl_read_write, access::target::local> sstd(ColsPerBlk, cgh);
    cgh.parallel_for<class sopKernel<Type, IdxType, TPB, ColsPerBlk>>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      int tx = item.get_local_id(1);
      int bx = item.get_group(1);
      int by = item.get_group(0);
      int gridDim_x = item.get_group_range(1);
 
      const int RowsPerBlkPerIter = TPB / ColsPerBlk;
      IdxType thisColId = tx % ColsPerBlk;
      IdxType thisRowId = tx / ColsPerBlk;
      IdxType colId = thisColId + ((IdxType)by * ColsPerBlk);
      IdxType rowId = thisRowId + ((IdxType)bx * RowsPerBlkPerIter);
      Type thread_data = Type(0);
      const IdxType stride = RowsPerBlkPerIter * gridDim_x;
      for (IdxType i = rowId; i < N; i += stride) {
        Type val = (colId < D) ? data[i * D + colId] : Type(0);
        thread_data += val * val;
      }
      if (tx < ColsPerBlk) sstd[tx] = Type(0);
      item.barrier(access::fence_space::local_space);

      //atomicAdd(sstd + thisColId, thread_data);
      auto atomic_local = ext::oneapi::atomic_ref<Type, 
            ext::oneapi::memory_order::relaxed,
            ext::oneapi::memory_scope::work_group,
            access::address_space::local_space> (sstd[thisColId]);
          atomic_local.fetch_add(thread_data);

      item.barrier(access::fence_space::local_space);

      if (tx < ColsPerBlk) {
        // atomicAdd(std + colId, sstd[thisColId]);
        auto atomic_global = ext::oneapi::atomic_ref<Type, 
                             ext::oneapi::memory_order::relaxed,
                             ext::oneapi::memory_scope::device,
                             access::address_space::global_space> (std[colId]);
        atomic_global.fetch_add(sstd[thisColId]);
      }
    });
  });

  range<1> gws2 ((D+TPB-1)/TPB*TPB);
  range<1> lws2 (TPB);
  IdxType sampleSize = sample ? N-1 : N;
  q.submit([&] (handler &cgh) {
    auto std = d_std.template get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class sampleKernel<Type, IdxType>>(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
      IdxType i = item.get_global_id(0);
      if (i < D) std[i] = sycl::sqrt(std[i] / sampleSize);
    });
  });
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    printf("D: number of columns of data (must be a multiple of 32)\n");
    printf("N: number of rows of data (at least one row)\n");
    return 1;
  }
  int D = atoi(argv[1]); // columns must be a multiple of 32
  int N = atoi(argv[2]); // at least one row
  int repeat = atoi(argv[3]);

  bool sample = true;
  long inputSize = D * N;
  long inputSizeByte = inputSize * sizeof(float);
  float *data = (float*) malloc (inputSizeByte);

  // input data 
  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  // host and device results
  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);

  { // sycl scope 
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<const float, 1> d_data (data, inputSize);

  buffer<float, 1> d_std (std, outputSize);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  // execute kernels on a device
  for (int i = 0; i < repeat; i++)
    stddev(q, d_std, d_data, D, N, sample);

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

  }

  // verify
  stddev_ref(std_ref, data, D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref[i] - std[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  free(std_ref);
  free(std);
  free(data);
  return 0;
}

