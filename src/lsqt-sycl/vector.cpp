/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, and Ari Harju

    This file is part of GPUQT.

    GPUQT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUQT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUQT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <string.h>    // memcpy
#include "vector.h"
#define BLOCK_SIZE 256

#ifndef CPU_ONLY
void gpu_set_zero(
  sycl::nd_item<1> &item,
  const int number_of_elements,
  real* __restrict g_state_real,
  real* __restrict g_state_imag)
{
  int n = item.get_global_id(0);
  if (n < number_of_elements) {
    g_state_real[n] = 0;
    g_state_imag[n] = 0;
  }
}
#else
void cpu_set_zero(int number_of_elements, real* g_state_real, real* g_state_imag)
{
  for (int n = 0; n < number_of_elements; ++n) {
    g_state_real[n] = 0;
    g_state_imag[n] = 0;
  }
}
#endif

#ifndef CPU_ONLY
void Vector::initialize_gpu(int n)
{
  this->n = n;
  array_size = n * sizeof(real);
  real_part = sycl::malloc_device<real>(n, q);
  imag_part = sycl::malloc_device<real>(n, q);
}
#else
void Vector::initialize_cpu(int n)
{
  this->n = n;
  array_size = n * sizeof(real);
  real_part = new real[n];
  imag_part = new real[n];
}
#endif

Vector::Vector(int n)
{
#ifndef CPU_ONLY
  initialize_gpu(n);

  sycl::range<1> gws (((n - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    auto real_part_t = real_part;
    auto imag_part_t = imag_part;
    cgh.parallel_for<class reset>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      gpu_set_zero(item, n, real_part_t, imag_part_t);
    });
  });
#else
  initialize_cpu(n);
  cpu_set_zero(n, real_part, imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_copy_state(
  sycl::nd_item<1> &item,
  const int n,
  const real* __restrict in_real,
  const real* __restrict in_imag,
        real* __restrict out_real,
        real* __restrict out_imag)
{
  int i = item.get_global_id(0);
  if (i < n) {
    out_real[i] = in_real[i];
    out_imag[i] = in_imag[i];
  }
}
#else
void cpu_copy_state(int N, real* in_real, real* in_imag, real* out_real, real* out_imag)
{
  for (int n = 0; n < N; ++n) {
    out_real[n] = in_real[n];
    out_imag[n] = in_imag[n];
  }
}
#endif

Vector::Vector(Vector& original)
{
  // Just teach myself: one can access private members of another instance
  // of the class from within the class
#ifndef CPU_ONLY
  const int size = original.n;  // implicit capture of 'this'(i.e. n) is not allowed for kernel functions
  initialize_gpu(size);

  sycl::range<1> gws (((size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    auto real_part_dst = real_part;
    auto imag_part_dst = imag_part;
    auto real_part_src = original.real_part;
    auto imag_part_src = original.imag_part;
    cgh.parallel_for<class copy>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      gpu_copy_state(item, size, real_part_src, imag_part_src,
                     real_part_dst, imag_part_dst);
    });
  });
#else
  initialize_cpu(original.n);
  cpu_copy_state(n, original.real_part, original.imag_part, real_part, imag_part);
#endif
}

Vector::~Vector()
{
#ifndef CPU_ONLY
  sycl::free(real_part, q);
  sycl::free(imag_part, q);
#else
  delete[] real_part;
  delete[] imag_part;
#endif
}

#ifndef CPU_ONLY
void gpu_add_state(
  sycl::nd_item<1> &item,
  const int n,
  const real*__restrict in_real,
  const real*__restrict in_imag,
        real*__restrict out_real,
        real*__restrict out_imag)
{
  int i = item.get_global_id(0);
  if (i < n) {
    out_real[i] += in_real[i];
    out_imag[i] += in_imag[i];
  }
}
#else
void cpu_add_state(int n, real* in_real, real* in_imag, real* out_real, real* out_imag)
{
  for (int i = 0; i < n; ++i) {
    out_real[i] += in_real[i];
    out_imag[i] += in_imag[i];
  }
}
#endif

void Vector::add(Vector& other)
{
#ifndef CPU_ONLY
  const int size = n;

  sycl::range<1> gws (((size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    auto real_part_dst = real_part;
    auto imag_part_dst = imag_part;
    auto real_part_src = other.real_part;
    auto imag_part_src = other.imag_part;
    cgh.parallel_for<class add2>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      gpu_add_state(
        item,
        size, real_part_src, imag_part_src,
        real_part_dst, imag_part_dst);
    });
  });
#else
  cpu_add_state(n, other.real_part, other.imag_part, real_part, imag_part);
#endif
}

void Vector::copy(Vector& other)
{
#ifndef CPU_ONLY
  const int size = n;
  sycl::range<1> gws (((size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    auto real_part_dst = real_part;
    auto imag_part_dst = imag_part;
    auto real_part_src = other.real_part;
    auto imag_part_src = other.imag_part;
    cgh.parallel_for<class copy2>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      gpu_copy_state(item, size, real_part_src, imag_part_src,
                     real_part_dst, imag_part_dst);
    });
  });
#else
  cpu_copy_state(n, other.real_part, other.imag_part, real_part, imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_apply_sz(
  sycl::nd_item<1> &item,
  const int n,
  const real* __restrict in_real,
  const real* __restrict in_imag,
        real* __restrict out_real,
        real* __restrict out_imag)
{
  int i = item.get_global_id(0);
  if (i < n) {
    if (i % 2 == 0) {
      out_real[i] = in_real[i];
      out_imag[i] = in_imag[i];
    } else {
      out_real[i] = -in_real[i];
      out_imag[i] = -in_imag[i];
    }
  }
}
#else
void cpu_apply_sz(int n, real* in_real, real* in_imag, real* out_real, real* out_imag)
{
  for (int i = 0; i < n; ++i) {
    if (i % 2 == 0) {
      out_real[i] = in_real[i];
      out_imag[i] = in_imag[i];
    } else {
      out_real[i] = -in_real[i];
      out_imag[i] = -in_imag[i];
    }
  }
}
#endif

void Vector::apply_sz(Vector& other)
{
#ifndef CPU_ONLY
  const int size = n;
  sycl::range<1> gws (((size - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    auto real_part_dst = real_part;
    auto imag_part_dst = imag_part;
    auto real_part_src = other.real_part;
    auto imag_part_src = other.imag_part;
    cgh.parallel_for<class apply_sz>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      gpu_apply_sz(item, size, real_part_src, imag_part_src,
                     real_part_dst, imag_part_dst);
    });
  });
#else
  cpu_apply_sz(n, other.real_part, other.imag_part, real_part, imag_part);
#endif
}

void Vector::copy_from_host(real* other_real, real* other_imag)
{
#ifndef CPU_ONLY
  q.memcpy(real_part, other_real, array_size);
  q.memcpy(imag_part, other_imag, array_size);
#else
  memcpy(real_part, other_real, array_size);
  memcpy(imag_part, other_imag, array_size);
#endif
}

void Vector::copy_to_host(real* target_real, real* target_imag)
{
#ifndef CPU_ONLY
  q.memcpy(target_real, real_part, array_size);
  q.memcpy(target_imag, imag_part, array_size);
  q.wait();
#else
  memcpy(target_real, real_part, array_size);
  memcpy(target_imag, imag_part, array_size);
#endif
}

void Vector::swap(Vector& other)
{
  real* tmp_real = real_part;
  real* tmp_imag = imag_part;
  real_part = other.real_part, imag_part = other.imag_part;
  other.real_part = tmp_real;
  other.imag_part = tmp_imag;
}

#ifndef CPU_ONLY
void warp_reduce(volatile real* s, int t)
{
  s[t] += s[t + 32];
  s[t] += s[t + 16];
  s[t] += s[t + 8];
  s[t] += s[t + 4];
  s[t] += s[t + 2];
  s[t] += s[t + 1];
}
#endif

#ifndef CPU_ONLY
void gpu_find_inner_product_1(
  sycl::nd_item<1> &item,
  const int number_of_atoms,
  const real* __restrict g_final_state_real,
  const real* __restrict g_final_state_imag,
  const real* __restrict g_random_state_real,
  const real* __restrict g_random_state_imag,
        real* __restrict g_inner_product_real,
        real* __restrict g_inner_product_imag,
        real* s_data_real,
        real* s_data_imag,
  const int g_offset)
{
  int tid = item.get_local_id(0);
  int bid = item.get_group(0);
  int n = item.get_global_id(0);
  int m;
  real a, b, c, d;
  s_data_real[tid] = 0.0;
  s_data_imag[tid] = 0.0;

  if (n < number_of_atoms) {
    a = g_final_state_real[n];
    b = g_final_state_imag[n];
    c = g_random_state_real[n];
    d = g_random_state_imag[n];
    s_data_real[tid] = (a * c + b * d);
    s_data_imag[tid] = (b * c - a * d);
  }
  item.barrier(sycl::access::fence_space::local_space);

/*
  if (tid < 256) {
    m = tid + 256;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  item.barrier(sycl::access::fence_space::local_space);
*/
  if (tid < 128) {
    m = tid + 128;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (tid < 64) {
    m = tid + 64;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (tid < 32) {
    warp_reduce(s_data_real, tid);
    warp_reduce(s_data_imag, tid);
  }
  if (tid == 0) {
    g_inner_product_real[bid + g_offset] = s_data_real[0];
    g_inner_product_imag[bid + g_offset] = s_data_imag[0];
  }
}
#else
void cpu_find_inner_product_1(
  int grid_size,
  int number_of_atoms,
  real* g_final_state_real,
  real* g_final_state_imag,
  real* g_random_state_real,
  real* g_random_state_imag,
  real* g_inner_product_real,
  real* g_inner_product_imag,
  int g_offset)
{
  for (int m = 0; m < grid_size; ++m) {
    real s_data_real = 0.0;
    real s_data_imag = 0.0;
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      int n = m * BLOCK_SIZE + k;
      if (n < number_of_atoms) {
        real a = g_final_state_real[n];
        real b = g_final_state_imag[n];
        real c = g_random_state_real[n];
        real d = g_random_state_imag[n];
        s_data_real += (a * c + b * d);
        s_data_imag += (b * c - a * d);
      }
    }
    g_inner_product_real[m + g_offset] = s_data_real;
    g_inner_product_imag[m + g_offset] = s_data_imag;
  }
}
#endif

void Vector::inner_product_1(int number_of_atoms, Vector& other, Vector& target, int offset)
{
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE + 1;
#ifndef CPU_ONLY
  sycl::range<1> gws (grid_size * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);
  q.submit([&] (sycl::handler &cgh) {
    auto real_part_t = real_part;
    auto imag_part_t = imag_part;
    auto other_real_part = other.real_part;
    auto other_imag_part = other.imag_part;
    auto target_real_part = target.real_part;
    auto target_imag_part = target.imag_part;
    sycl::local_accessor<real, 1> s_data_real(sycl::range<1>(BLOCK_SIZE), cgh);
    sycl::local_accessor<real, 1> s_data_imag(sycl::range<1>(BLOCK_SIZE), cgh);
    cgh.parallel_for<class dot1>(
     sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
     gpu_find_inner_product_1(
       item,
       number_of_atoms,
       real_part_t,
       imag_part_t,
       other_real_part,
       other_imag_part,
       target_real_part,
       target_imag_part,
       s_data_real.get_pointer(),
       s_data_imag.get_pointer(),
       offset);
    });
  });
#else
  cpu_find_inner_product_1(
    grid_size, number_of_atoms, real_part, imag_part, other.real_part, other.imag_part,
    target.real_part, target.imag_part, offset);
#endif
}

#ifndef CPU_ONLY
void gpu_find_inner_product_2(
  sycl::nd_item<1> &item,
  const int number_of_atoms,
  const real* __restrict g_inner_product_1_real,
  const real* __restrict g_inner_product_1_imag,
        real* __restrict g_inner_product_2_real,
        real* __restrict g_inner_product_2_imag,
        real* s_data_real,
        real* s_data_imag)
{
  int tid = item.get_local_id(0);
  int bid = item.get_group(0);
  int patch, n, m;

  s_data_real[tid] = 0.0;
  s_data_imag[tid] = 0.0;
  int number_of_blocks = (number_of_atoms - 1) / BLOCK_SIZE + 1;
  int number_of_patches = (number_of_blocks - 1) / BLOCK_SIZE + 1;

  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * BLOCK_SIZE;
    if (n < number_of_blocks) {
      m = bid * number_of_blocks + n;
      s_data_real[tid] += g_inner_product_1_real[m];
      s_data_imag[tid] += g_inner_product_1_imag[m];
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

/*
  if (tid < 256) {
    m = tid + 256;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  item.barrier(sycl::access::fence_space::local_space);
*/
  if (tid < 128) {
    m = tid + 128;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (tid < 64) {
    m = tid + 64;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (tid < 32) {
    warp_reduce(s_data_real, tid);
    warp_reduce(s_data_imag, tid);
  }
  if (tid == 0) {
    g_inner_product_2_real[bid] = s_data_real[0];
    g_inner_product_2_imag[bid] = s_data_imag[0];
  }
}
#else
void cpu_find_inner_product_2(
  int number_of_moments,
  int grid_size,
  real* g_inner_product_1_real,
  real* g_inner_product_1_imag,
  real* g_inner_product_2_real,
  real* g_inner_product_2_imag)
{
  for (int m = 0; m < number_of_moments; ++m) {
    real s_data_real = 0.0;
    real s_data_imag = 0.0;
    for (int k = 0; k < grid_size; ++k) {
      int n = m * grid_size + k;
      s_data_real += g_inner_product_1_real[n];
      s_data_imag += g_inner_product_1_imag[n];
    }
    g_inner_product_2_real[m] = s_data_real;
    g_inner_product_2_imag[m] = s_data_imag;
  }
}
#endif

void Vector::inner_product_2(int number_of_atoms, int number_of_moments, Vector& target)
{
#ifndef CPU_ONLY
  sycl::range<1> gws (number_of_moments * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    auto real_part_dst = target.real_part;
    auto imag_part_dst = target.imag_part;
    auto real_part_src = real_part;
    auto imag_part_src = imag_part;
    sycl::local_accessor<real, 1> s_data_real(sycl::range<1>(BLOCK_SIZE), cgh);
    sycl::local_accessor<real, 1> s_data_imag(sycl::range<1>(BLOCK_SIZE), cgh);
    cgh.parallel_for<class dot2>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      gpu_find_inner_product_2(
        item,
        number_of_atoms,
        real_part_src,
        imag_part_src,
        real_part_dst,
        imag_part_dst,
        s_data_real.get_pointer(),
        s_data_imag.get_pointer());
    });
  });
#else
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE + 1;
  cpu_find_inner_product_2(
    number_of_moments, grid_size, real_part, imag_part, target.real_part, target.imag_part);
#endif
}
