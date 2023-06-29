/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Tridiagonal solvers.
 * Device code for sweep solver (one-system-per-thread).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */


// solves a bunch of tridiagonal linear systems
// much better performance when doing data reordering before
// so that all memory accesses are coalesced
using float4 = sycl::float4;

void sweep_small_systems_local_kernel(
    sycl::nd_item<1> &item,
    const float * a_d,
    const float * b_d,
    const float * c_d,
    const float * d_d,
    float * x_d,
    int system_size,
    int num_systems,
    bool reorder)
{
  int i = item.get_global_id(0);

  // need to check for in-bounds because of the thread block size
  if (i >= num_systems) return;

  int stride = reorder ? num_systems: 1;
  int base_idx = reorder ? i : i * system_size;

  // local memory
  float a[128];

  float c1, c2, c3;
  float f_i, x_prev, x_next;

  // solving next system:
  // c1 * u_i+1 + c2 * u_i + c3 * u_i-1 = f_i

  c1 = c_d[base_idx];
  c2 = b_d[base_idx];
  f_i = d_d[base_idx];

#ifndef NATIVE_DIVIDE
  a[1] = - c1 / c2;
  x_prev = f_i / c2;
#else
  a[1] = - sycl::native::divide(c1, c2);
  x_prev = sycl::native::divide(f_i, c2);
#endif

  // forward trace
  int idx = base_idx;
  x_d[base_idx] = x_prev;
  for (int k = 1; k < system_size-1; k++)
  {
    idx += stride;

    c1 = c_d[idx];
    c2 = b_d[idx];
    c3 = a_d[idx];
    f_i = d_d[idx];

    float q = (c3 * a[k] + c2);
#ifndef NATIVE_DIVIDE
    float t = 1 / q;
#else
    float t = sycl::native::recip(q);
#endif
    x_next = (f_i - c3 * x_prev) * t;
    x_d[idx] = x_prev = x_next;

    a[k+1] = - c1 * t;
  }

  idx += stride;

  c2 = b_d[idx];
  c3 = a_d[idx];
  f_i = d_d[idx];

  float q = (c3 * a[system_size-1] + c2);
#ifndef NATIVE_DIVIDE
  float t = 1 / q;
#else
  float t = sycl::native::recip(q);
#endif
  x_next = (f_i - c3 * x_prev) * t;
  x_d[idx] = x_prev = x_next;

  // backward trace
  for (int k = system_size-2; k >= 0; k--)
  {
    idx -= stride;
    x_next = x_d[idx];
    x_next += x_prev * a[k+1];
    x_d[idx] = x_prev = x_next;
  }
}

inline int getLocalIdx(int i, int k, int num_systems)
{
  return i + num_systems * k;

  // uncomment for uncoalesced mem access
  // return k + system_size * i;
}

void sweep_small_systems_global_kernel(
    sycl::nd_item<1> &item,
    const float * a_d,
    const float * b_d,
    const float * c_d,
    const float * d_d,
    float * x_d,
    float * w_d,
    int system_size,
    int num_systems,
    bool reorder)
{
  int i = item.get_global_id(0);

  // need to check for in-bounds because of the thread block size
  if (i >= num_systems) return;

  int stride = reorder ? num_systems: 1;
  int base_idx = reorder ? i : i * system_size;

  float c1, c2, c3;
  float f_i, x_prev, x_next;

  // solving next system:
  // c1 * u_i+1 + c2 * u_i + c3 * u_i-1 = f_i

  c1 = c_d[base_idx];
  c2 = b_d[base_idx];
  f_i = d_d[base_idx];

#ifndef NATIVE_DIVIDE
  w_d[getLocalIdx(i, 1, num_systems)] = - c1 / c2;
  x_prev = f_i / c2;
#else
  w_d[getLocalIdx(i, 1, num_systems)] = - sycl::native::divide(c1, c2);
  x_prev = sycl::native::divide(f_i, c2);
#endif

  // forward trace
  int idx = base_idx;
  x_d[base_idx] = x_prev;
  for (int k = 1; k < system_size-1; k++)
  {
    idx += stride;

    c1 = c_d[idx];
    c2 = b_d[idx];
    c3 = a_d[idx];
    f_i = d_d[idx];

    float q = (c3 * w_d[getLocalIdx(i, k, num_systems)] + c2);
#ifndef NATIVE_DIVIDE
    float t = 1 / q;
#else
    float t = sycl::native::recip(q);
#endif
    x_next = (f_i - c3 * x_prev) * t;
    x_d[idx] = x_prev = x_next;

    w_d[getLocalIdx(i, k+1, num_systems)] = - c1 * t;
  }

  idx += stride;

  c2 = b_d[idx];
  c3 = a_d[idx];
  f_i = d_d[idx];

  float q = (c3 * w_d[getLocalIdx(i, system_size-1, num_systems)] + c2);
#ifndef NATIVE_DIVIDE
  float t = 1 / q;
#else
  float t = sycl::native::recip(q);
#endif
  x_next = (f_i - c3 * x_prev) * t;
  x_d[idx] = x_prev = x_next;

  // backward trace
  for (int k = system_size-2; k >= 0; k--)
  {
    idx -= stride;
    x_next = x_d[idx];
    x_next += x_prev * w_d[getLocalIdx(i, k+1, num_systems)];
    x_d[idx] = x_prev = x_next;
  }
}

inline float4 load(const float * a, int i)
{
  return float4(a[i], a[i+1], a[i+2], a[i+3]);
}

inline void store(float * a, int i, float4 v)
{
  a[i] = v.x();
  a[i+1] = v.y();
  a[i+2] = v.z();
  a[i+3] = v.w();
}

void sweep_small_systems_global_vec4_kernel(
    sycl::nd_item<1> &item,
    const float * a_d,
    const float * b_d,
    const float * c_d,
    const float * d_d,
    float * x_d,
    float * w_d,
    int system_size,
    int num_systems,
    bool reorder)
{
  int j = item.get_global_id(0);
  int i = j << 2;

  // need to check for in-bounds because of the thread block size
  if (i >= num_systems) return;

  int stride = reorder ? num_systems: 4;
  int base_idx = reorder ? i : i * system_size;

  float4 c1, c2, c3;
  float4 f_i, x_prev, x_next;

  // solving next system:
  // c1 * u_i+1 + c2 * u_i + c3 * u_i-1 = f_i

  c1 = load(c_d, base_idx);
  c2 = load(b_d, base_idx);
  f_i = load(d_d, base_idx);

#ifndef NATIVE_DIVIDE
  store(w_d, getLocalIdx(i, 1, num_systems), - c1 / c2);
  x_prev = f_i / c2;
#else
  store(w_d, getLocalIdx(i, 1, num_systems), - sycl::native::divide(c1, c2));
  x_prev = sycl::native::divide(f_i, c2);
#endif

  // forward trace
  int idx = base_idx;
  store(x_d, base_idx, x_prev);
  for (int k = 1; k < system_size-1; k++)
  {
    idx += stride;

    c1 = load(c_d, idx);
    c2 = load(b_d, idx);
    c3 = load(a_d, idx);
    f_i = load(d_d, idx);

    float4 q = (c3 * load(w_d, getLocalIdx(i, k, num_systems)) + c2);
#ifndef NATIVE_DIVIDE
    float4 t = float4(1,1,1,1) / q;
#else
    float4 t = sycl::native::recip(q);
#endif
    x_next = (f_i - c3 * x_prev) * t;
    x_prev = x_next;
    store(x_d, idx, x_prev);

    store(w_d, getLocalIdx(i, k+1, num_systems), - c1 * t);
  }

  idx += stride;

  c2 = load(b_d, idx);
  c3 = load(a_d, idx);
  f_i = load(d_d, idx);

  float4 q = (c3 * load(w_d, getLocalIdx(i, system_size-1, num_systems)) + c2);
#ifndef NATIVE_DIVIDE
  float4 t = float4(1,1,1,1) / q;
#else
  float4 t = sycl::native::recip(q);
#endif
  x_next = (f_i - c3 * x_prev) * t;
  x_prev = x_next;
  store(x_d, idx, x_prev);

  // backward trace
  for (int k = system_size-2; k >= 0; k--)
  {
    idx -= stride;
    x_next = load(x_d, idx);
    x_next += x_prev * load(w_d, getLocalIdx(i, k+1, num_systems));
    x_prev = x_next;
    store(x_d, idx, x_prev);
  }
}

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory
// so that bank conflicts do not occur when threads address the array column-wise.
void transpose(
    sycl::nd_item<2> &item,
    float * odata,
    const float * idata,
    float * block,
    int width,
    int height)
{
  int blockIdxx = item.get_group(1);
  int blockIdxy = item.get_group(0);

  int threadIdxx = item.get_local_id(1);
  int threadIdxy = item.get_local_id(0);

  // evaluate coordinates and check bounds
  int i0 = sycl::mul24(blockIdxx, BLOCK_DIM) + threadIdxx;
  int j0 = sycl::mul24(blockIdxy, BLOCK_DIM) + threadIdxy;

  if (i0 >= width || j0 >= height) return;

  int i1 = sycl::mul24(blockIdxy, BLOCK_DIM) + threadIdxx;
  int j1 = sycl::mul24(blockIdxx, BLOCK_DIM) + threadIdxy;

  if (i1 >= height || j1 >= width) return;

  int idx_a = i0 + sycl::mul24(j0, width);
  int idx_b = i1 + sycl::mul24(j1, height);

  // read the tile from global memory into shared memory
  block[threadIdxy * (BLOCK_DIM+1) + threadIdxx] = idata[idx_a];

  item.barrier(sycl::access::fence_space::local_space);

  // write back to transposed array
  odata[idx_b] = block[threadIdxx * (BLOCK_DIM+1) + threadIdxy];
}
