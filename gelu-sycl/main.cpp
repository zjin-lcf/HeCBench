#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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

// width is hidden_dim and height is seq_len
void gelu_bias_loop(sycl::half *src, const sycl::half *bias, int width,
                    int height, sycl::nd_item<2> &item)
{
  int batch = item.get_group(1);
  int x = item.get_group(0); // seq length
  int y = item.get_local_id(1) * 2;

  if (x < height) {
    int    index = batch * width * height + x * width;
    sycl::half2 v_src;
    sycl::half2 v_bias;
    sycl::half2 v;
    sycl::float2 t;
    for (; y < width; y = y + item.get_local_range(1) * 2) {
      v_bias = ((sycl::half2 *)bias)[y >> 1];
      v_src = ((sycl::half2 *)src)[(index + y) >> 1];
      v = v_src + v_bias;
      t = v.convert<float, sycl::rounding_mode::automatic>();
      t.x() = (0.5f * t.x() * (1.0f + sycl::tanh(0.79788456f * (t.x() + 0.044715f * t.x() * t.x() * t.x()))));
      t.y() = (0.5f * t.y() * (1.0f + sycl::tanh(0.79788456f * (t.y() + 0.044715f * t.y() * t.y() * t.y()))));

      ((sycl::half2 *)src)[(index + y) >> 1] = t.convert<sycl::half, sycl::rounding_mode::rte>();
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <batch> <sequence length> <hidden dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int batch_size = atoi(argv[1]);
  const int seq_len = atoi(argv[2]);
  const int hidden_dim = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const size_t src_size = (size_t)batch_size * seq_len * hidden_dim;

  const size_t src_size_bytes = src_size * sizeof(sycl::half);
  const int bias_size_bytes = hidden_dim * sizeof(sycl::half);

  srand(123);
  sycl::half *output = (sycl::half *)malloc(src_size_bytes);
  for (size_t i = 0; i < src_size; i++) {
    output[i] = sycl::vec<float, 1>{rand() / (float)RAND_MAX}
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }

  sycl::half *bias = (sycl::half *)malloc(bias_size_bytes);
  for (int i = 0; i < hidden_dim; i++) {
    bias[i] = sycl::vec<float, 1>{-6 + (rand() % 12)}
                  .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::half *d_output;
  d_output = (sycl::half *)sycl::malloc_device(src_size_bytes, q);
  q.memcpy(d_output, output, src_size_bytes);

  sycl::half *d_bias;
  d_bias = (sycl::half *)sycl::malloc_device(bias_size_bytes, q);
  q.memcpy(d_bias, bias, bias_size_bytes);

  sycl::range<2> lws (1, 1024);
  sycl::range<2> gws (seq_len, batch_size * 1024);

  q.wait(); 
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class gelu>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        gelu_bias_loop(d_output, d_bias, hidden_dim, seq_len, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(output, d_output, src_size_bytes).wait();

  float sum = 0;
  for (size_t i = 0; i < src_size; i++) {
    sum += sycl::vec<sycl::half, 1>{output[i]}
               .convert<float, sycl::rounding_mode::automatic>()[0];
  }
  printf("Checksum = %f\n", sum / src_size);

  sycl::free(d_output, q);
  sycl::free(d_bias, q);
  free(output);
  free(bias);

  return 0;
}
