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
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

const int F = 9;
const int THREAD_M = 16;

template <unsigned int WarpSize>
__global__ void Code1x16MatVec(
    const int4* __restrict__ A, const int4* __restrict__ B,
    int4* __restrict__ C, const int4* __restrict__ codebook, const int prob_m,
    const int prob_k,
    const int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                  // codebook, at most 3 long.
    const int codebook_stride     // as int4.
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (blockDim.x / WarpSize) * blockIdx.x + (threadIdx.x / WarpSize);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x;
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % WarpSize;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % WarpSize;

  // We pad shared memory to avoid bank conflicts during reads
  __shared__ int4 sh_b[WarpSize * F];
  float res = 0;

  int iters = (prob_k / 8 + 8 * WarpSize - 1) / (8 * WarpSize);
  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < WarpSize * 8; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) sh_b[F * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    __syncthreads();
    b_gl_rd += WarpSize * 8;

    int b_sh_rd = F * (threadIdx.x % WarpSize);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[4];
        int4 t = codebook[enc[i]];
        dec[0] = t.x;
        dec[1] = t.y;
        dec[2] = t.z;
        dec[3] = t.w;

        half2* a = reinterpret_cast<half2*>(&dec);
        half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++) res2 = __hfma2(a[j], b[j], res2);
        res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd++;
      }
      a_gl_rd += WarpSize;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = WarpSize/2; i > 0; i /= 2) res += __shfl_down(res, i);
    if (threadIdx.x % WarpSize == 0) {
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
    }
  }
}

void code1x16_matvec(const void* __restrict__ A,
                     const void* __restrict__ B, void* __restrict__ C,
                     const void* __restrict__ codebook, int prob_m,
                     int prob_k, const int4 codebook_a_sizes,
                     const int codebook_stride) {
  int sms;
  hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);

  int WarpSize;
  hipDeviceGetAttribute(&WarpSize, hipDeviceAttributeWarpSize, 0);
  int threads = WarpSize * thread_m;

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  if (WarpSize == 64)
    Code1x16MatVec<64><<<blocks, threads, sizeof(int4) * WarpSize * F, 0>>>(
        (const int4*)A, (const int4*)B, (int4*)C, (const int4*)codebook, prob_m,
        prob_k, codebook_a_sizes, codebook_stride);
  else
    Code1x16MatVec<32><<<blocks, threads, sizeof(int4) * WarpSize * F, 0>>>(
        (const int4*)A, (const int4*)B, (int4*)C, (const int4*)codebook, prob_m,
        prob_k, codebook_a_sizes, codebook_stride);

  hipDeviceSynchronize();
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
  int4 codebook_a_sizes = make_int4(4096, 8192, 12288, 122880);
  int codebook_stride = entries * 8 * sizeof(__half) / sizeof(int4);

  //size_t scale_size = cols;

  size_t input_size_bytes = input_size * sizeof(__half); 
  size_t output_size_bytes = output_size * sizeof(__half); 
  size_t code_size_bytes = code_size * sizeof(short); 
  size_t codebook_size_bytes = codebook_size * sizeof(__half); 
  //size_t scale_size_bytes = scale_size * sizeof(__half); 

  __half *h_input, *h_output, *h_codebook;
  //__half *h_scale;
  short *h_codes;

  h_input = (__half*) malloc (input_size_bytes);
  h_output = (__half*) malloc (output_size_bytes);
  h_codes = (short*) malloc (code_size_bytes);
  h_codebook = (__half*) malloc (codebook_size_bytes);
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

  __half *d_input, *d_output, *d_codebook;
  //__half *d_scale;
  short *d_codes;

  hipMalloc(&d_input, input_size_bytes);
  hipMalloc(&d_output, output_size_bytes);
  hipMalloc(&d_codes, code_size_bytes);
  hipMalloc(&d_codebook, codebook_size_bytes);
  //hipMalloc(&d_scale, scale_size_bytes);

  hipMemcpy(d_input, h_input, input_size_bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_codes, h_codes, code_size_bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_codebook, h_codebook, codebook_size_bytes, hipMemcpyHostToDevice);

  int prob_m = 12288;
  int prob_k = 4096;

  for (int i = 0; i < b; ++i) {
    auto d_input_vec = d_input + 4096 * i;
    auto d_output_vec = d_output + 12288/32 * i;
    code1x16_matvec((void*)d_codes, (void*)d_input_vec,
                    (void*)d_output_vec, (void*)d_codebook,
                    prob_m, prob_k, codebook_a_sizes, codebook_stride);
  }

  hipMemcpy(h_output, d_output, output_size_bytes, hipMemcpyDeviceToHost);

#ifdef DEBUG
  for (int i = 0; i < b; ++i) {
    auto h_output_vec = h_output + 12288/32 * i;
    for (int j = 0; j < 12288/32; j++) {
      printf("row=%d col=%d val=%f\n", i, j, __half2float(h_output_vec[j]));
    }
  }
#endif

  hipFree(d_input);
  hipFree(d_output);
  hipFree(d_codes);
  hipFree(d_codebook);
  //hipFree(d_scale);
  free(h_input);
  free(h_output);
  free(h_codes);
  free(h_codebook);
  //free(h_scale);
}
