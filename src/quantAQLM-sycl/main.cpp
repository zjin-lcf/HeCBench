/*
 * Modified by Neural Magic
 * Adapted from https://github.com/Vahe1994/AQLM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

const int F = 9;
const int THREAD_M = 16;

void Code1x16MatVec(
    const sycl::int4 *__restrict__ A, const sycl::int4 *__restrict__ B,
    sycl::int4 *__restrict__ C, const sycl::int4 *__restrict__ codebook,
    const int prob_m, const int prob_k,
    const sycl::int4 codebook_a_sizes, // cumulative sizes of A spanning each
                                       // codebook, at most 3 long.
    const int codebook_stride,
    const sycl::nd_item<3> &item,
    sycl::int4 *sh_b // as int4.
) {

  auto sg = item.get_sub_group();
  int warpSize = sg.get_max_local_range().get(0);

  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = sg.get_group_linear_range() * item.get_group(2) + sg.get_group_linear_id();
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x();
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + sg.get_local_linear_id();
  int a_gl_end = a_gl_rd + a_gl_stride - sg.get_local_linear_id();

  float res = 0;

  int iters = (prob_k / 8 + 8 * warpSize - 1) / (8 * warpSize);
  while (iters--) {
    item.barrier(sycl::access::fence_space::local_space);
    for (int i = item.get_local_id(2); i < warpSize * 8; i += item.get_local_range(2)) {
      if (b_gl_rd + i < prob_k / 8) sh_b[F * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    item.barrier(sycl::access::fence_space::local_space);
    b_gl_rd += warpSize * 8;

    int b_sh_rd = F * sg.get_local_linear_id();
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[4];
        sycl::int4 t = codebook[enc[i]];
        dec[0] = t.x();
        dec[1] = t.y();
        dec[2] = t.z();
        dec[3] = t.w();

        sycl::half2 *a = reinterpret_cast<sycl::half2 *>(&dec);
        sycl::half2 *b = reinterpret_cast<sycl::half2 *>(&sh_b[b_sh_rd]);
        sycl::half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++) res2 = sycl::fma(a[j], b[j], res2);
        res += float(res2.x()) + float(res2.y());
        b_sh_rd++;
      }
      a_gl_rd += warpSize;
    }
  }

  if (pred) {
    res = sycl::reduce_over_group(sg, res, sycl::plus<float>()); 
    if (sg.leader()) {
      reinterpret_cast<sycl::half *>(C)[c_gl_wr] = sycl::half(res);
    }
  }
}

void code1x16_matvec(sycl::queue &q,
                     const void *__restrict__ A, const void *__restrict__ B,
                     void *__restrict__ C, const void *__restrict__ codebook,
                     int prob_m, int prob_k, const sycl::int4 codebook_a_sizes,
                     const int codebook_stride) {
  int sms;
  sms = q.get_device().get_info<sycl::info::device::max_compute_units>();
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int warpSize = *r;

  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = warpSize * thread_m;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<sycl::int4, 1> sh_b_acc(
          sycl::range<1>(warpSize * F), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, blocks * threads),
                            sycl::range<3>(1, 1, threads)),
          [=](sycl::nd_item<3> item) {
            Code1x16MatVec(
                (const sycl::int4 *)A, (const sycl::int4 *)B, (sycl::int4 *)C,
                (const sycl::int4 *)codebook, prob_m, prob_k, codebook_a_sizes,
                codebook_stride, item,
                sh_b_acc.get_multi_ptr<sycl::access::decorated::no>().get());
      });
  }).wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel execution time: %f (us)\n", (time * 1e-3f) / 1);
}

int main() {

  const int b = 4;
  
  // size of each input vector is 4096
  size_t input_size = b * 4096;

  // weight matrix  X input vectors
  size_t output_size = b * 12288 / 32;

  // weight matrix
  size_t rows = 512;  // 4096 / sizeof(int4)
  size_t cols = 12288;
  size_t code_size = cols * rows;

  // codebook
  size_t entries = 65536;
  size_t codebook_size = 3 * entries * 8;
  //int codebook_partition_sizes[] = {4096, 4096, 4096};
  sycl::int4 codebook_a_sizes = sycl::int4(4096, 8192, 12288, 122880);
  int codebook_stride = entries * 8 * sizeof(sycl::half) / sizeof(sycl::int4);

  //size_t scale_size = cols;

  size_t input_size_bytes = input_size * sizeof(sycl::half);
  size_t output_size_bytes = output_size * sizeof(sycl::half);
  size_t code_size_bytes = code_size * sizeof(short);
  size_t codebook_size_bytes = codebook_size * sizeof(sycl::half);
  //size_t scale_size_bytes = scale_size * sizeof(sycl::half);

  sycl::half *h_input, *h_output, *h_codebook;
  //__half *h_scale;
  short *h_codes;

  h_input = (sycl::half *)malloc(input_size_bytes);
  h_output = (sycl::half *)malloc(output_size_bytes);
  h_codes = (short*) malloc (code_size_bytes);
  h_codebook = (sycl::half *)malloc(codebook_size_bytes);
  //h_scale = (__half*) malloc (scale_size_bytes);

  srand(123);
  for (size_t i = 0; i < input_size; i++)  
    h_input[i] = (float)i / input_size;

  for (size_t i = 0; i < code_size; i++)  
    h_codes[i] = rand() % 65536 - 32768;

  for (size_t i = 0; i < codebook_size; i++)  
    h_codebook[i] = (float)i / codebook_size;

  //for (size_t i = 0; i < scale_size; i++)  
  //  h_scale[i] = (float)i / scale_size;

  sycl::property_list p {sycl::property::queue::in_order()};
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, p);
#else
  sycl::queue q(sycl::cpu_selector_v, p);
#endif

  sycl::half *d_input, *d_output, *d_codebook;
  //sycl::half *d_scale;
  short *d_codes;

  d_input = (sycl::half *)sycl::malloc_device(input_size_bytes, q);
  d_output = (sycl::half *)sycl::malloc_device(output_size_bytes, q);
  d_codes = (short *)sycl::malloc_device(code_size_bytes, q);
  d_codebook = (sycl::half *)sycl::malloc_device(codebook_size_bytes, q);
  //d_scale = (sycl::half *)sycl::malloc_device(scale_size_bytes, q);

  q.memcpy(d_input, h_input, input_size_bytes);
  q.memcpy(d_codes, h_codes, code_size_bytes);
  q.memcpy(d_codebook, h_codebook, codebook_size_bytes);

  int prob_m = 12288;
  int prob_k = 4096;

  for (int i = 0; i < b; ++i) {
    auto d_input_vec = d_input + 4096 * i;
    auto d_output_vec = d_output + 12288/32 * i;
    code1x16_matvec(q,
                    (void*)d_codes, (void*)d_input_vec,
                    (void*)d_output_vec, (void*)d_codebook,
                    prob_m, prob_k, codebook_a_sizes, codebook_stride);
  }

  q.memcpy(h_output, d_output, output_size_bytes).wait();

#ifdef DEBUG
  for (int i = 0; i < b; ++i) {
    auto h_output_vec = h_output + 12288/32 * i;
    for (int j = 0; j < 12288/32; j++) {
      printf("row=%d col=%d val=%f\n", i, j, float(h_output_vec[j]));
    }
  }
#endif

  sycl::free(d_input, q);
  sycl::free(d_output, q);
  sycl::free(d_codes, q);
  sycl::free(d_codebook, q);
  //sycl::free(d_scale, q);
 
  free(h_input);
  free(h_output);
  free(h_codes);
  free(h_codebook);
  //free(h_scale);
}
