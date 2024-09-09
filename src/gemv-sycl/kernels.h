#include <stdio.h>
#include <sycl/sycl.hpp>

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

struct uint4_2 {
  uint8_t data;

  uint4_2(uint8_t x = 0, uint8_t y = 0) {
    setX(x);
    setY(y);
  }

  uint8_t getX() const {
    return data & 0x0F;  // get the lower 4 bits
  }

  uint8_t getY() const {
    return (data >> 4) & 0x0F;  // get the upper 4 bits
  }

  void setX(uint8_t x) {
    data = (data & 0xF0) | (x & 0x0F);  // set the lower 4 bits
  }

  void setY(uint8_t y) {
    data = (data & 0x0F) | ((y & 0x0F) << 4);  // set the upper 4 bits
  }
};

struct __half4 {
  sycl::half x, y, z, w;
};
struct int8_2 {
  int8_t x, y;
};
struct uint4_2_4 {
  uint4_2 x, y, z, w;
};

///////////////////////////// REDUCE SUM //////////////////////////////

inline float warpReduceSum(float sum, unsigned int threadNum,
                           const sycl::nd_item<3> &item) {

  auto sg = item.get_sub_group();

  if (threadNum >= 32)
    sum += sycl::shift_group_left(sg, sum, 16); // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += sycl::shift_group_left(sg, sum, 8); // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += sycl::shift_group_left(sg, sum, 4); // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += sycl::shift_group_left(sg, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += sycl::shift_group_left(sg, sum, 1); // 0-1, 2-3, 4-5, etc.
  return sum;
}

///////////////////////////// NORMAL //////////////////////////////
// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
void gemv_fp16(
    sycl::half *mat, sycl::half *vec, sycl::half *res, unsigned int n,
    unsigned int num_per_thread, const sycl::nd_item<3> &item,
    float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE]) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = item.get_local_id(2);
  unsigned int row = item.get_group(1) * item.get_local_range(1) +
                     item.get_local_id(1);
  unsigned int start_idx = item.get_local_id(2);
  sycl::float4 *mat4 = reinterpret_cast<sycl::float4 *>(mat);
  sycl::float4 *vec4 = reinterpret_cast<sycl::float4 *>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * item.get_local_range(2);
    if (j < n >> 3) {
      sycl::float4 vec_val = vec4[j];
      sycl::float4 mat_val = mat4[row * (n >> 3) + j];
      const sycl::half2 *vec_h1 = (sycl::half2 *)&vec_val.x();
      const sycl::half2 *vec_h2 = (sycl::half2 *)&vec_val.y();
      const sycl::half2 *vec_h3 = (sycl::half2 *)&vec_val.z();
      const sycl::half2 *vec_h4 = (sycl::half2 *)&vec_val.w();
      const sycl::half2 *mat_h1 = (sycl::half2 *)&mat_val.x();
      const sycl::half2 *mat_h2 = (sycl::half2 *)&mat_val.y();
      const sycl::half2 *mat_h3 = (sycl::half2 *)&mat_val.z();
      const sycl::half2 *mat_h4 = (sycl::half2 *)&mat_val.w();
      sum += static_cast<float>(vec_h1->x()) * static_cast<float>(mat_h1->x());
      sum += static_cast<float>(vec_h1->y()) * static_cast<float>(mat_h1->y());
      sum += static_cast<float>(vec_h2->x()) * static_cast<float>(mat_h2->x());
      sum += static_cast<float>(vec_h2->y()) * static_cast<float>(mat_h2->y());
      sum += static_cast<float>(vec_h3->x()) * static_cast<float>(mat_h3->x());
      sum += static_cast<float>(vec_h3->y()) * static_cast<float>(mat_h3->y());
      sum += static_cast<float>(vec_h4->x()) * static_cast<float>(mat_h4->x());
      sum += static_cast<float>(vec_h4->y()) * static_cast<float>(mat_h4->y());
    }
  }

  sum = warpReduceSum(sum, item.get_local_range(2), item);

  if (item.get_local_range(2) <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = sycl::vec<float, 1>(sum)
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)

  const int laneId = item.get_local_id(2) % WARP_SIZE;
  const int warpId = item.get_local_id(2) / WARP_SIZE;
  if (laneId == 0) warpLevelSums[item.get_local_id(1)][warpId] = sum;
  item.barrier(sycl::access::fence_space::local_space);
  // read from shared memory only if that warp existed
  sum = (item.get_local_id(2) < item.get_local_range(2) / WARP_SIZE)
            ? warpLevelSums[item.get_local_id(1)][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum =
      warpReduceSum(sum, item.get_local_range(2) / WARP_SIZE, item);
  if (tid == 0) {
    res[row] = sycl::vec<float, 1>(sum)
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
}

///////////////////////////// QUANTIZED-INT8 //////////////////////////////

void gemv_quantized_int8(
    int8_t *mat, sycl::half *vec, sycl::half *res, unsigned int n,
    sycl::half scale, sycl::half zero_point, unsigned int num_per_thread,
    const sycl::nd_item<3> &item,
    float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE]) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = item.get_local_id(2);
  unsigned int row = item.get_group(1) * item.get_local_range(1) +
                     item.get_local_id(1);
  unsigned int start_idx = item.get_local_id(2);
  __half4* mat4 = reinterpret_cast<__half4*>(mat);
  sycl::float4 *vec4 = reinterpret_cast<sycl::float4 *>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * item.get_local_range(2);
    if (j < n >> 3) {
      sycl::float4 vec_val = vec4[j];
      __half4 mat_val = mat4[row * (n >> 3) + j];
      const sycl::half2 *vec_h1 = (sycl::half2 *)&vec_val.x();
      const sycl::half2 *vec_h2 = (sycl::half2 *)&vec_val.y();
      const sycl::half2 *vec_h3 = (sycl::half2 *)&vec_val.z();
      const sycl::half2 *vec_h4 = (sycl::half2 *)&vec_val.w();
      const int8_2* mat_h1 = (int8_2*)&mat_val.x;
      const int8_2* mat_h2 = (int8_2*)&mat_val.y;
      const int8_2* mat_h3 = (int8_2*)&mat_val.z;
      const int8_2* mat_h4 = (int8_2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x()) *
             (static_cast<float>(mat_h1->x) - zero_point_f);
      sum += static_cast<float>(vec_h1->y()) *
             (static_cast<float>(mat_h1->y) - zero_point_f);
      sum += static_cast<float>(vec_h2->x()) *
             (static_cast<float>(mat_h2->x) - zero_point_f);
      sum += static_cast<float>(vec_h2->y()) *
             (static_cast<float>(mat_h2->y) - zero_point_f);
      sum += static_cast<float>(vec_h3->x()) *
             (static_cast<float>(mat_h3->x) - zero_point_f);
      sum += static_cast<float>(vec_h3->y()) *
             (static_cast<float>(mat_h3->y) - zero_point_f);
      sum += static_cast<float>(vec_h4->x()) *
             (static_cast<float>(mat_h4->x) - zero_point_f);
      sum += static_cast<float>(vec_h4->y()) *
             (static_cast<float>(mat_h4->y) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, item.get_local_range(2), item);

  if (item.get_local_range(2) <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = sycl::vec<float, 1>(sum)
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)

  const int laneId = item.get_local_id(2) % WARP_SIZE;
  const int warpId = item.get_local_id(2) / WARP_SIZE;
  if (laneId == 0) warpLevelSums[item.get_local_id(1)][warpId] = sum;
  item.barrier(sycl::access::fence_space::local_space);
  // read from shared memory only if that warp existed
  sum = (item.get_local_id(2) < item.get_local_range(2) / WARP_SIZE)
            ? warpLevelSums[item.get_local_id(1)][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum =
      warpReduceSum(sum, item.get_local_range(2) / WARP_SIZE, item);
  if (tid == 0) {
    res[row] = sycl::vec<float, 1>(sum)
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
}

///////////////////////////// QUANTIZED-INT4 //////////////////////////////

// based on previous experiments, num_per_thread can >= 16
void gemv_quantized_int4(
    uint4_2 *mat, sycl::half *vec, sycl::half *res, unsigned int n,
    sycl::half scale, sycl::half zero_point, unsigned int num_per_thread,
    const sycl::nd_item<3> &item,
    float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE]) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = item.get_local_id(2);
  unsigned int row = item.get_group(1) * item.get_local_range(1) +
                     item.get_local_id(1);
  unsigned int start_idx = item.get_local_id(2);
  uint4_2_4* mat4 = reinterpret_cast<uint4_2_4*>(mat);
  sycl::float4 *vec4 = reinterpret_cast<sycl::float4 *>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 4; iter++) {
    unsigned int j = 2 * (start_idx + iter * item.get_local_range(2));
    if (j < n >> 3) {
      sycl::float4 vec_val_1 = vec4[j]; // 8 __half
      sycl::float4 vec_val_2 = vec4[j + 1];
      const sycl::half2 *vec_h1 = (sycl::half2 *)&vec_val_1.x();
      const sycl::half2 *vec_h2 = (sycl::half2 *)&vec_val_1.y();
      const sycl::half2 *vec_h3 = (sycl::half2 *)&vec_val_1.z();
      const sycl::half2 *vec_h4 = (sycl::half2 *)&vec_val_1.w();
      const sycl::half2 *vec_h5 = (sycl::half2 *)&vec_val_2.x();
      const sycl::half2 *vec_h6 = (sycl::half2 *)&vec_val_2.y();
      const sycl::half2 *vec_h7 = (sycl::half2 *)&vec_val_2.z();
      const sycl::half2 *vec_h8 = (sycl::half2 *)&vec_val_2.w();

      uint4_2_4 mat_val_1 = mat4[row * (n >> 3) + j];
      uint4_2_4 mat_val_2 = mat4[row * (n >> 3) + j + 1];
      const uint4_2* mat_h1 = (uint4_2*)&mat_val_1.x;
      const uint4_2* mat_h2 = (uint4_2*)&mat_val_1.y;
      const uint4_2* mat_h3 = (uint4_2*)&mat_val_1.z;
      const uint4_2* mat_h4 = (uint4_2*)&mat_val_1.w;
      const uint4_2* mat_h5 = (uint4_2*)&mat_val_2.x;
      const uint4_2* mat_h6 = (uint4_2*)&mat_val_2.y;
      const uint4_2* mat_h7 = (uint4_2*)&mat_val_2.z;
      const uint4_2* mat_h8 = (uint4_2*)&mat_val_2.w;

      sum += static_cast<float>(vec_h1->x()) *
             (static_cast<float>(mat_h1->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h1->y()) *
             (static_cast<float>(mat_h1->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h2->x()) *
             (static_cast<float>(mat_h2->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h2->y()) *
             (static_cast<float>(mat_h2->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h3->x()) *
             (static_cast<float>(mat_h3->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h3->y()) *
             (static_cast<float>(mat_h3->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h4->x()) *
             (static_cast<float>(mat_h4->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h4->y()) *
             (static_cast<float>(mat_h4->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h5->x()) *
             (static_cast<float>(mat_h5->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h5->y()) *
             (static_cast<float>(mat_h5->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h6->x()) *
             (static_cast<float>(mat_h6->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h6->y()) *
             (static_cast<float>(mat_h6->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h7->x()) *
             (static_cast<float>(mat_h7->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h7->y()) *
             (static_cast<float>(mat_h7->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h8->x()) *
             (static_cast<float>(mat_h8->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h8->y()) *
             (static_cast<float>(mat_h8->getY()) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, item.get_local_range(2), item);

  if (item.get_local_range(2) <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = sycl::vec<float, 1>(sum)
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)

  const int laneId = item.get_local_id(2) % WARP_SIZE;
  const int warpId = item.get_local_id(2) / WARP_SIZE;
  if (laneId == 0) warpLevelSums[item.get_local_id(1)][warpId] = sum;
  item.barrier(sycl::access::fence_space::local_space);
  // read from shared memory only if that warp existed
  sum = (item.get_local_id(2) < item.get_local_range(2) / WARP_SIZE)
            ? warpLevelSums[item.get_local_id(1)][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum =
      warpReduceSum(sum, item.get_local_range(2) / WARP_SIZE, item);
  if (tid == 0) {
    res[row] = sycl::vec<float, 1>(sum)
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
}

