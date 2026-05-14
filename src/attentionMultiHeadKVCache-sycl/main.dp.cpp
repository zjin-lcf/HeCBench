/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
#include <type_traits>
#include "cpu_reference.h"
#include "flashinfer_ops.dp.hpp"
#include "utils.h"

using namespace flashinfer;

template <typename DTypeQO, typename DTypeKV>
void _TestDecodingKernelCorrectness(sycl::queue &que,
                                    int repeat, size_t num_qo_heads,
                                    size_t num_kv_heads, size_t seq_len,
                                    size_t head_dim, QKVLayout kv_layout,
                                    PosEncodingMode pos_encoding_mode) try {
  size_t QO_size_bytes = num_qo_heads * head_dim * sizeof(DTypeQO);
  size_t KV_size_bytes = seq_len * num_kv_heads * head_dim * sizeof(DTypeKV);
  std::vector<DTypeQO> Q_host(num_qo_heads * head_dim);
  std::vector<DTypeKV> K_host(seq_len * num_kv_heads * head_dim);
  std::vector<DTypeKV> V_host(seq_len * num_kv_heads * head_dim);
  std::vector<DTypeQO> O_host(num_qo_heads * head_dim);

  // initialize data with normal distribution
  utils::vec_normal_(Q_host);
  utils::vec_normal_(K_host);
  utils::vec_normal_(V_host);

  DTypeQO *Q, *O, *tmp;
  DTypeKV *K, *V;

  Q = (DTypeQO*) sycl::malloc_device(QO_size_bytes, que);
  O = (DTypeQO*) sycl::malloc_device(QO_size_bytes, que);
  tmp = (DTypeQO*) sycl::malloc_device(sizeof(DTypeQO) * 32 * 1024 * 1024, que);
  K = (DTypeKV*) sycl::malloc_device(KV_size_bytes, que);
  V = (DTypeKV*) sycl::malloc_device(KV_size_bytes, que);

  que.memcpy(Q, Q_host.data(), QO_size_bytes);
  que.memcpy(K, K_host.data(), KV_size_bytes);
  que.memcpy(V, V_host.data(), KV_size_bytes);

  std::vector<DTypeQO> O_ref_host;

  O_ref_host = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
      Q_host, K_host, V_host, 1, seq_len, num_qo_heads, num_kv_heads, head_dim, false, kv_layout,
      pos_encoding_mode);

  SingleDecodeWithKVCache<DTypeQO, DTypeKV, DTypeQO>(
    que, Q, K, V, O, tmp, num_qo_heads, num_kv_heads, seq_len, head_dim,
    kv_layout, pos_encoding_mode);

  que.memcpy(O_host.data(), O, QO_size_bytes).wait();

  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < num_qo_heads * head_dim; ++i) {
    if (std::isnan(float(O_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(O_host[i]), float(O_ref_host[i]), 1e-2, 1e-2));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(num_qo_heads * head_dim);
  std::cout << "num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
            << ", seq_len=" << seq_len << ", head_dim=" << head_dim
            << ", kv_layout=" << QKVLayoutToString(kv_layout)
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result accuracy (atol=1e-3, rtol=1e-3)=" << result_accuracy;
  if (result_accuracy < 0.90)
    std::cerr << " FAIL ";
  else
    std::cerr << " PASS ";
  if (nan_detected) std::cerr << "NaN detected.";
  std::cout << std::endl;

  que.wait();
  auto start = std::chrono::steady_clock::now();
  
  for (int i = 0; i < repeat; i++) {
    SingleDecodeWithKVCache<DTypeQO, DTypeKV, DTypeQO>(
      que, Q, K, V, O, tmp, num_qo_heads, num_kv_heads, seq_len, head_dim,
      kv_layout, pos_encoding_mode);
  }
  que.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleDecodeWithKVCache: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(Q, que);
  sycl::free(K, que);
  sycl::free(V, que);
  sycl::free(O, que);
  sycl::free(tmp, que);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename DTypeQO, typename DTypeKV>
void SingleDecodeKernel(sycl::queue &que, const int repeat) {
  for (size_t num_qo_heads : {32}) {
    for (size_t num_kv_heads : {4, 8, 32}) {
      for (size_t seq_len : {1, 3, 9, 27, 81, 129, 257, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (unsigned int kv_layout : {0U, 1U}) {
            for (unsigned int pos_encoding_mode : {0U, 1U}) {
              _TestDecodingKernelCorrectness<DTypeQO, DTypeKV>(que, repeat, num_qo_heads, num_kv_heads, seq_len,
                                                               head_dim, QKVLayout(kv_layout),
                                                               PosEncodingMode(pos_encoding_mode));
            }
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue que(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue que(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::cout << "FlashInfer: SingleDecodeKernel (FP16)" << std::endl;
  SingleDecodeKernel<sycl::half,
                     sycl::half>(que, repeat);

  std::cout << std::endl;

  std::cout << "FlashInfer: SingleDecodeKernel (BF16)" << std::endl;
  SingleDecodeKernel<sycl::ext::oneapi::bfloat16,
                     sycl::ext::oneapi::bfloat16>(que, repeat);

  return 0;
}
