/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define BUF_SIZE 256
#define PATTERN 0xDEADBEEF

#define CUDACHECK(code)                                                         \
  do {                                                                         \
    cudaerr = code;                                                             \
    if (cudaerr != cudaSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)cudaerr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, 
    unsigned int numGroups, unsigned int subGroupSize) {
  for (unsigned i = 0; i < numGroups; ++i) {
    for (unsigned j = 0; j < subGroupSize; j++) {
      output[i * subGroupSize + j] = input[i * subGroupSize + subGroupSize - j - 1];
    }
  }
}

void verifyBroadcast(const int *out, const int subGroupSize, int pattern = 0)
{
  int expected = pattern;
  if (pattern == 0) {
    for (int i = 0; i < subGroupSize; i++) 
      expected += i;
  }
  int errors = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != expected) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
      break;
    }
  }
  if (errors == 0) std::cout << "PASSED\n";
}

void verifyTransposeMatrix(const float *TransposeMatrix, const float* cpuTransposeMatrix, 
            const int total, const int subGroupSize)
{
  int errors = 0;
  float eps = 1.0E-6;
  for (int i = 0; i < total; i++) {
    if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      std::cout << "(sg" << subGroupSize << ") ";
      std::cout << "ITEM: " << i <<
        " cpu: " << cpuTransposeMatrix[i] <<
        " gpu: " << TransposeMatrix[i] << "\n";
      errors++;
      break;
    }
  }
  if (errors == 0) std::cout << "PASSED\n";
}


//==================================================================================
// Broadcast
//==================================================================================
void bcast_shfl_sg8(const int arg, int *out, sycl::nd_item<3> item_ct1) {
  int value = ((item_ct1.get_local_id(2) & 0x7) == 0) ? arg : 0;
  // Synchronize all threads in warp, and get "value" from lane 0
  int out_v = item_ct1.get_sub_group().shuffle(value, 0);
  size_t oi = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  out[oi] = out_v;
}

void bcast_shfl_xor_sg8(int *out, sycl::nd_item<3> item_ct1) {
  int value = (item_ct1.get_local_id(2) & 0x7);
  for (int mask = 1; mask < 0x7; mask *= 2)
    value += item_ct1.get_sub_group().shuffle_xor(value, mask);
  size_t oi = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  out[oi] = value;
}

void bcast_shfl_sg16(const int arg, int *out, sycl::nd_item<3> item_ct1) {
  int value = ((item_ct1.get_local_id(2) & 0xf) == 0) ? arg : 0;
  // Synchronize all threads in warp, and get "value" from lane 0
  int out_v = item_ct1.get_sub_group().shuffle(value, 0);
  size_t oi = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  out[oi] = out_v;
}

void bcast_shfl_xor_sg16(int *out, sycl::nd_item<3> item_ct1) {
  int value = (item_ct1.get_local_id(2) & 0xf);
  for (int mask = 1; mask < 0xf; mask *= 2)
    value += item_ct1.get_sub_group().shuffle_xor(value, mask);
  size_t oi = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  out[oi] = value;
}
void bcast_shfl_sg32(const int arg, int *out, sycl::nd_item<3> item_ct1) {
  int value = ((item_ct1.get_local_id(2) & 0x1f) == 0) ? arg : 0;
  // Synchronize all threads in warp, and get "value" from lane 0
  int out_v = item_ct1.get_sub_group().shuffle(value, 0);
  size_t oi = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  out[oi] = out_v;
}

void bcast_shfl_xor_sg32(int *out, sycl::nd_item<3> item_ct1) {
  int value = (item_ct1.get_local_id(2) & 0x1f);
  for (int mask = 1; mask < 0x1f; mask *= 2)
    value += item_ct1.get_sub_group().shuffle_xor(value, mask);
  size_t oi = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  out[oi] = value;
}
//==================================================================================
// Matrix transpose
//==================================================================================
void transpose_shfl(float* out, const float* in, sycl::nd_item<3> item_ct1) {
  unsigned b_start = item_ct1.get_local_range().get(2) * item_ct1.get_group(2);
  unsigned b_offs = b_start + item_ct1.get_local_id(2);
  unsigned s_offs =
      item_ct1.get_local_range().get(2) - item_ct1.get_local_id(2) - 1;
  float val = in[b_offs];
  out[b_offs] = item_ct1.get_sub_group().shuffle(val, s_offs);
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  std::cout << "Broadcast using shuffle functions\n";

  int *out = (int *)malloc(sizeof(int) * BUF_SIZE);
  int *d_out;
  dpct::dpct_malloc((void **)&d_out, sizeof(int) * BUF_SIZE);

  std::cout << "Broadcast using the shuffle xor function (subgroup sizes 8, 16, and 32) \n";
  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t d_out_buf_ct0 = dpct::get_buffer(d_out);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_out_acc_ct0 =
          d_out_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BUF_SIZE),
                                         sycl::range<3>(1, 1, BUF_SIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         bcast_shfl_xor_sg8((int *)(&d_out_acc_ct0[0]),
                                            item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(out, d_out, sizeof(int) * BUF_SIZE, dpct::device_to_host);
  verifyBroadcast(out, 8);

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t d_out_buf_ct0 = dpct::get_buffer(d_out);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_out_acc_ct0 =
          d_out_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BUF_SIZE),
                                         sycl::range<3>(1, 1, BUF_SIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         bcast_shfl_xor_sg16((int *)(&d_out_acc_ct0[0]),
                                             item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(out, d_out, sizeof(int) * BUF_SIZE, dpct::device_to_host);
  verifyBroadcast(out, 16);

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t d_out_buf_ct0 = dpct::get_buffer(d_out);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_out_acc_ct0 =
          d_out_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BUF_SIZE),
                                         sycl::range<3>(1, 1, BUF_SIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         bcast_shfl_xor_sg32((int *)(&d_out_acc_ct0[0]),
                                             item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(out, d_out, sizeof(int) * BUF_SIZE, dpct::device_to_host);
  verifyBroadcast(out, 32);

  std::cout << "Broadcast using the shuffle function (subgroup sizes 8, 16, and 32) \n";

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t d_out_buf_ct1 = dpct::get_buffer(d_out);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_out_acc_ct1 =
          d_out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BUF_SIZE),
                                         sycl::range<3>(1, 1, BUF_SIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         bcast_shfl_sg8(PATTERN, (int *)(&d_out_acc_ct1[0]),
                                        item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(out, d_out, sizeof(int) * BUF_SIZE, dpct::device_to_host);
  verifyBroadcast(out, 8, PATTERN);

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t d_out_buf_ct1 = dpct::get_buffer(d_out);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_out_acc_ct1 =
          d_out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BUF_SIZE),
                                         sycl::range<3>(1, 1, BUF_SIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         bcast_shfl_sg16(PATTERN, (int *)(&d_out_acc_ct1[0]),
                                         item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(out, d_out, sizeof(int) * BUF_SIZE, dpct::device_to_host);
  verifyBroadcast(out, 16, PATTERN);

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t d_out_buf_ct1 = dpct::get_buffer(d_out);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_out_acc_ct1 =
          d_out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BUF_SIZE),
                                         sycl::range<3>(1, 1, BUF_SIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         bcast_shfl_sg32(PATTERN, (int *)(&d_out_acc_ct1[0]),
                                         item_ct1);
                       });
    });
  }
  dpct::dpct_memcpy(out, d_out, sizeof(int) * BUF_SIZE, dpct::device_to_host);
  verifyBroadcast(out, 32, PATTERN);

  free(out);
  dpct::dpct_free(d_out);

  std::cout << "matrix transpose using the shuffle function (subgroup sizes are 8, 16, and 32)\n";

  const int total = 1 << 27;  // total number of elements in a matrix

  float* Matrix = (float*)malloc(total * sizeof(float));
  float* TransposeMatrix = (float*)malloc(total * sizeof(float));
  float* cpuTransposeMatrix = (float*)malloc(total * sizeof(float));

  // initialize the input data
  for (int i = 0; i < total; i++) {
    Matrix[i] = (float)i * 10.0f;
  }

  float *gpuMatrix;
  float *gpuTransposeMatrix;
  // allocate the memory on the device side
  dpct::dpct_malloc((void **)&gpuMatrix, total * sizeof(float));
  dpct::dpct_malloc((void **)&gpuTransposeMatrix, total * sizeof(float));

  dpct::dpct_memcpy(gpuMatrix, Matrix, total * sizeof(float),
                    dpct::host_to_device);

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t gpuTransposeMatrix_buf_ct0 =
        dpct::get_buffer(gpuTransposeMatrix);
    dpct::buffer_t gpuMatrix_buf_ct1 = dpct::get_buffer(gpuMatrix);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto gpuTransposeMatrix_acc_ct0 =
          gpuTransposeMatrix_buf_ct0.get_access<sycl::access::mode::read_write>(
              cgh);
      auto gpuMatrix_acc_ct1 =
          gpuMatrix_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, total / 8) *
                                             sycl::range<3>(1, 1, 8),
                                         sycl::range<3>(1, 1, 8)),
                       [=](sycl::nd_item<3> item_ct1) {
                         transpose_shfl(
                             (float *)(&gpuTransposeMatrix_acc_ct0[0]),
                             (const float *)(&gpuMatrix_acc_ct1[0]), item_ct1);
                       });
    });
  }

  // Memory transfer from device to host
  dpct::dpct_memcpy(TransposeMatrix, gpuTransposeMatrix, total * sizeof(float),
                    dpct::device_to_host);
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/8, 8);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 8);

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t gpuTransposeMatrix_buf_ct0 =
        dpct::get_buffer(gpuTransposeMatrix);
    dpct::buffer_t gpuMatrix_buf_ct1 = dpct::get_buffer(gpuMatrix);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto gpuTransposeMatrix_acc_ct0 =
          gpuTransposeMatrix_buf_ct0.get_access<sycl::access::mode::read_write>(
              cgh);
      auto gpuMatrix_acc_ct1 =
          gpuMatrix_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, total / 16) *
                                             sycl::range<3>(1, 1, 16),
                                         sycl::range<3>(1, 1, 16)),
                       [=](sycl::nd_item<3> item_ct1) {
                         transpose_shfl(
                             (float *)(&gpuTransposeMatrix_acc_ct0[0]),
                             (const float *)(&gpuMatrix_acc_ct1[0]), item_ct1);
                       });
    });
  }

  // Memory transfer from device to host
  dpct::dpct_memcpy(TransposeMatrix, gpuTransposeMatrix, total * sizeof(float),
                    dpct::device_to_host);
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/16, 16);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 16);

  for (int n = 0; n < 100; n++)
  {
    dpct::buffer_t gpuTransposeMatrix_buf_ct0 =
        dpct::get_buffer(gpuTransposeMatrix);
    dpct::buffer_t gpuMatrix_buf_ct1 = dpct::get_buffer(gpuMatrix);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto gpuTransposeMatrix_acc_ct0 =
          gpuTransposeMatrix_buf_ct0.get_access<sycl::access::mode::read_write>(
              cgh);
      auto gpuMatrix_acc_ct1 =
          gpuMatrix_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, total / 32) *
                                             sycl::range<3>(1, 1, 32),
                                         sycl::range<3>(1, 1, 32)),
                       [=](sycl::nd_item<3> item_ct1) {
                         transpose_shfl(
                             (float *)(&gpuTransposeMatrix_acc_ct0[0]),
                             (const float *)(&gpuMatrix_acc_ct1[0]), item_ct1);
                       });
    });
  }

  // Memory transfer from device to host
  dpct::dpct_memcpy(TransposeMatrix, gpuTransposeMatrix, total * sizeof(float),
                    dpct::device_to_host);
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, total/32, 32);
  verifyTransposeMatrix(TransposeMatrix, cpuTransposeMatrix, total, 32);


  // free the resources
  dpct::dpct_free(gpuMatrix);
  dpct::dpct_free(gpuTransposeMatrix);

  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  return 0;
}
