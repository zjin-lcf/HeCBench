#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include "kernels.h"
#include "simple_tensor.h"

///////////////////////////// SOLVER //////////////////////////////

SimpleTensor<sycl::half> solve_gemv_int4_quantized_with_params(
    sycl::queue &q,
    const int repeat, const SimpleTensor<uint4_2> &mat,
    const SimpleTensor<sycl::half> &vec, unsigned int block_dim_x,
    unsigned int block_dim_y, float scale_f, float zero_point_f)
{
  sycl::half scale =
      sycl::vec<float, 1>(scale_f)
          .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  sycl::half zero_point =
      sycl::vec<float, 1>(zero_point_f)
          .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  assert(mat.width_ * 2 == vec.height_);
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = vec.height_ / block_dim_x;
  assert(num_per_thread >= 16);
  SimpleTensor<sycl::half> result(q, vec.height_, 1);
  sycl::range<3> gws (1, mat.height_, block_dim_x);
  sycl::range<3> lws (1, block_dim_y, block_dim_x);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float[SHARED_MEM_MAX_ROWS][WARP_SIZE], 0>
          warpLevelSums_acc(cgh);

      auto mat_data_ = mat.data_;
      auto vec_data_ = vec.data_;
      auto result_data_ = result.data_;
      auto vec_height_ = vec.height_;

      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            gemv_quantized_int4(mat_data_, vec_data_, result_data_,
                                vec_height_, scale, zero_point,
                                num_per_thread, item,
                                warpLevelSums_acc);
          });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  return result;
}

SimpleTensor<sycl::half> solve_gemv_int8_quantized_with_params(
    sycl::queue &q,
    const int repeat, const SimpleTensor<int8_t> &mat,
    const SimpleTensor<sycl::half> &vec, unsigned int block_dim_x,
    unsigned int block_dim_y, float scale_f, float zero_point_f)
{
  sycl::half scale =
      sycl::vec<float, 1>(scale_f)
          .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  sycl::half zero_point =
      sycl::vec<float, 1>(zero_point_f)
          .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  assert(mat.width_ == vec.height_);
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = mat.width_ / block_dim_x;
  assert(num_per_thread >= 8);
  SimpleTensor<sycl::half> result(q, vec.height_, 1);

  sycl::range<3> gws (1, mat.height_, block_dim_x);
  sycl::range<3> lws (1, block_dim_y, block_dim_x);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float[SHARED_MEM_MAX_ROWS][WARP_SIZE], 0>
          warpLevelSums_acc(cgh);

      auto mat_data_ = mat.data_;
      auto vec_data_ = vec.data_;
      auto result_data_ = result.data_;
      auto mat_width_ = mat.width_;

      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            gemv_quantized_int8(mat_data_, vec_data_, result_data_,
                                mat_width_, scale, zero_point,
                                num_per_thread, item,
                                warpLevelSums_acc);
          });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  return result;
}

SimpleTensor<sycl::half>
solve_gemv_with_params(sycl::queue &q,
                       const int repeat, const SimpleTensor<sycl::half> &mat,
                       const SimpleTensor<sycl::half> &vec,
                       unsigned int block_dim_x, unsigned int block_dim_y)
{
  assert(mat.width_ == vec.height_);
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = mat.width_ / block_dim_x;
  assert(num_per_thread >= 8);
  SimpleTensor<sycl::half> result(q, vec.height_, 1);
  sycl::range<3> gws (1, mat.height_, block_dim_x);
  sycl::range<3> lws (1, block_dim_y, block_dim_x);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float[SHARED_MEM_MAX_ROWS][WARP_SIZE], 0>
          warpLevelSums_acc(cgh);

      auto mat_data_ = mat.data_;
      auto vec_data_ = vec.data_;
      auto result_data_ = result.data_;
      auto mat_width_ = mat.width_;

      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            gemv_fp16(mat_data_, vec_data_, result_data_,
                      mat_width_, num_per_thread, item,
                      warpLevelSums_acc);
          });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  return result;
}

void check_correctness(sycl::half *mat, sycl::half *vec, sycl::half *res, int n,
                       const sycl::nd_item<3> &item,
                       const sycl::stream &stream) {
  int idx = item.get_group(2) * item.get_local_range(2) +
            item.get_local_id(2);
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      result += sycl::vec<sycl::half, 1>(mat[idx * n + j])
                    .convert<float, sycl::rounding_mode::automatic>()[0] *
                sycl::vec<sycl::half, 1>(vec[j])
                    .convert<float, sycl::rounding_mode::automatic>()[0];
    }
    float diff =
        result - sycl::vec<sycl::half, 1>(res[idx])
                     .convert<float, sycl::rounding_mode::automatic>()[0];
    float delta = 0.125 * n / 512;
    if (diff > delta || diff < -delta) {
      stream << "idx= " << idx << ", diff = " << diff << "\n";
    }
  }
}

void check_int8_quantized_correctness(int8_t *mat, sycl::half *vec,
                                      sycl::half *res, sycl::half scale,
                                      sycl::half zero_point, int n,
                                      const sycl::nd_item<3> &item,
                                      const sycl::stream &stream) {
  int idx = item.get_group(2) * item.get_local_range(2) +
            item.get_local_id(2);
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      float dequantized_val = (static_cast<float>(mat[idx * n + j]) -
                               static_cast<float>(zero_point)) *
                              static_cast<float>(scale);
      result += dequantized_val *
                sycl::vec<sycl::half, 1>(vec[j])
                    .convert<float, sycl::rounding_mode::automatic>()[0];
    }
    float diff =
        result - sycl::vec<sycl::half, 1>(res[idx])
                     .convert<float, sycl::rounding_mode::automatic>()[0];
    float delta = 0.125 * n / 512;
    if (diff > delta || diff < -delta) {
      stream << "idx= " << idx << ", diff = " << diff << "\n";
    }
  }
}

void check_int4_quantized_correctness(uint4_2 *mat, sycl::half *vec,
                                      sycl::half *res, sycl::half scale,
                                      sycl::half zero_point, int mat_size,
                                      const sycl::nd_item<3> &item,
                                      const sycl::stream &stream) {
  int idx = item.get_group(2) * item.get_local_range(2) +
            item.get_local_id(2);
  if (idx < mat_size * 2) {
    float result = 0;
    for (int j = 0; j < mat_size; ++j) {
      uint8_t x = mat[idx * mat_size + j].getX();
      uint8_t y = mat[idx * mat_size + j].getY();
      float dequantized_x =
          (static_cast<float>(x) - static_cast<float>(zero_point)) *
          static_cast<float>(scale);
      float dequantized_y =
          (static_cast<float>(y) - static_cast<float>(zero_point)) *
          static_cast<float>(scale);
      result += dequantized_x *
                sycl::vec<sycl::half, 1>(vec[j * 2])
                    .convert<float, sycl::rounding_mode::automatic>()[0];
      result += dequantized_y *
                sycl::vec<sycl::half, 1>(vec[j * 2 + 1])
                    .convert<float, sycl::rounding_mode::automatic>()[0];
    }
    float diff =
        result - sycl::vec<sycl::half, 1>(res[idx])
                     .convert<float, sycl::rounding_mode::automatic>()[0];
    float delta = 0.125 * mat_size / 256;
    if (diff > delta || diff < -delta) {
      stream << "idx= " << idx << ", diff = " << diff << "\n";
    }
  }
}

void test_gemv_int4_quantized_with_params(sycl::queue &q,
                                          unsigned int size, unsigned int repeat,
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, float scale,
                                          float zero_point) {
  // generate data
  const unsigned int mat_width = size / 2;
  SimpleTensor<uint4_2> mat(q, size, mat_width);
  SimpleTensor<sycl::half> vec(q, size, 1);
  mat.reset();
  vec.reset();

  // compute dot product
  printf("GEMV int4 quantized...\n");
  SimpleTensor<sycl::half> res(q, size, 1);

  res = solve_gemv_int4_quantized_with_params(q, repeat, mat, vec, block_dim_x,
                                              block_dim_y, scale, zero_point);

  // check correctness
  printf("Check correctness on the device\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  
  q.submit([&](sycl::handler &cgh) {
    sycl::stream stream(64 * 1024, 80, cgh);

    auto mat_device_data = mat.device_data();
    auto vec_device_data = vec.device_data();
    auto res_device_data = res.device_data();

    cgh.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item) {
          check_int4_quantized_correctness(
              mat_device_data, vec_device_data, res_device_data,
              scale, zero_point, mat_width, item, stream);
        });
  }).wait();
}

void test_gemv_int8_quantized_with_params(sycl::queue &q,
                                          unsigned int size, unsigned int repeat,
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, float scale,
                                          float zero_point) {
  // generate data
  SimpleTensor<int8_t> mat(q, size, size);
  SimpleTensor<sycl::half> vec(q, size, 1);
  mat.reset();
  vec.reset();

  // compute the dot product
  printf("GEMV int8 quantized...\n");
  SimpleTensor<sycl::half> res(q, size, 1);

  res = solve_gemv_int8_quantized_with_params(q, repeat, mat, vec, block_dim_x,
                                              block_dim_y, scale, zero_point);

  // check correctness
  printf("Check correctness on the device\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  q.submit([&](sycl::handler &cgh) {
    sycl::stream stream(64 * 1024, 80, cgh);

    auto mat_device_data = mat.device_data();
    auto vec_device_data = vec.device_data();
    auto res_device_data = res.device_data();

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                        sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item) {
          check_int8_quantized_correctness(
              mat_device_data, vec_device_data, res_device_data,
              scale, zero_point, size, item, stream);
        });
  }).wait();
}

void test_gemv_with_params(sycl::queue &q,
                           unsigned int size, unsigned int repeat,
                           unsigned int block_dim_x, unsigned int block_dim_y) {
  // generate data
  SimpleTensor<sycl::half> mat(q, size, size);
  SimpleTensor<sycl::half> vec(q, size, 1);
  mat.reset();
  vec.reset();

  // compute the dot product
  printf("GEMV FP16 ..\n");
  SimpleTensor<sycl::half> res(q, size, 1);

  res = solve_gemv_with_params(q, repeat, mat, vec, block_dim_x, block_dim_y);

  // check correctness
  printf("Check correctness on the device\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  q.submit([&](sycl::handler &cgh) {
    sycl::stream stream(64 * 1024, 80, cgh);

    auto mat_device_data = mat.device_data();
    auto vec_device_data = vec.device_data();
    auto res_device_data = res.device_data();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                          sycl::range<3>(1, 1, threads_per_block)),
        [=](sycl::nd_item<3> item) {
          check_correctness(mat_device_data, vec_device_data,
                            res_device_data, size, item, stream);
        });
  }).wait();
}
