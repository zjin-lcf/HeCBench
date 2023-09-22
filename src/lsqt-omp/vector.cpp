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

#include "vector.h"
#include <string.h>    // memcpy
#define BLOCK_SIZE 256

#ifndef CPU_ONLY
void gpu_set_zero(int number_of_elements, 
  real* __restrict g_state_real, 
  real* __restrict g_state_imag)
{
  #pragma omp target teams distribute parallel for thread_limit (BLOCK_SIZE)
  for (int n = 0; n < number_of_elements; n++) {
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
  real_part = new real[n];
  imag_part = new real[n];
  #pragma omp target enter data map (alloc: real_part[0:n], imag_part[0:n])
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
  gpu_set_zero(n, real_part, imag_part);
#else
  initialize_cpu(n);
  cpu_set_zero(n, real_part, imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_copy_state(
  const int N,
  const real* __restrict in_real,
  const real* __restrict in_imag, 
        real* __restrict out_real, 
        real* __restrict out_imag)
{
  #pragma omp target teams distribute parallel for thread_limit (BLOCK_SIZE)
  for (int n = 0; n < N; n++) {
    out_real[n] = in_real[n];
    out_imag[n] = in_imag[n];
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
  initialize_gpu(original.n);
  gpu_copy_state(n, original.real_part, original.imag_part, real_part, imag_part);
#else
  initialize_cpu(original.n);
  cpu_copy_state(n, original.real_part, original.imag_part, real_part, imag_part);
#endif
}

Vector::~Vector()
{
#ifndef CPU_ONLY
  #pragma omp target exit data map (delete: real_part[0:n], imag_part[0:n])
  delete[] real_part;
  delete[] imag_part;
#else
  delete[] real_part;
  delete[] imag_part;
#endif
}

#ifndef CPU_ONLY
void gpu_add_state(
  const int n, 
  const real*__restrict in_real,
  const real*__restrict in_imag, 
        real*__restrict out_real, 
        real*__restrict out_imag)
{
  #pragma omp target teams distribute parallel for thread_limit (BLOCK_SIZE)
  for (int i = 0; i < n; i++) {
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
  gpu_add_state(n, other.real_part, other.imag_part, real_part, imag_part);
#else
  cpu_add_state(n, other.real_part, other.imag_part, real_part, imag_part);
#endif
}

void Vector::copy(Vector& other)
{
#ifndef CPU_ONLY
  gpu_copy_state(n, other.real_part, other.imag_part, real_part, imag_part);
#else
  cpu_copy_state(n, other.real_part, other.imag_part, real_part, imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_apply_sz(
  const int n, 
  const real* __restrict in_real, 
  const real* __restrict in_imag, 
        real* __restrict out_real, 
        real* __restrict out_imag)
{
  #pragma omp target teams distribute parallel for thread_limit (BLOCK_SIZE)
  for (int i = 0; i < n; i++) {
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
  gpu_apply_sz(n, other.real_part, other.imag_part, real_part, imag_part);
#else
  cpu_apply_sz(n, other.real_part, other.imag_part, real_part, imag_part);
#endif
}

void Vector::copy_from_host(real* other_real, real* other_imag)
{
#ifndef CPU_ONLY
 
  memcpy(real_part, other_real, array_size);
  memcpy(imag_part, other_imag, array_size);
  #pragma omp target update to (real_part[0:n])
  #pragma omp target update to (imag_part[0:n])
#else
  memcpy(real_part, other_real, array_size);
  memcpy(imag_part, other_imag, array_size);
#endif
}

void Vector::copy_to_host(real* target_real, real* target_imag)
{
#ifndef CPU_ONLY
  #pragma omp target update from (real_part[0:n])
  #pragma omp target update from (imag_part[0:n])
  memcpy(target_real, real_part, array_size);
  memcpy(target_imag, imag_part, array_size);
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
void gpu_find_inner_product_1(
  const int grid_size,
  const int number_of_atoms,
  const real* __restrict g_final_state_real,
  const real* __restrict g_final_state_imag,
  const real* __restrict g_random_state_real,
  const real* __restrict g_random_state_imag,
        real* __restrict g_inner_product_real,
        real* __restrict g_inner_product_imag,
  const int g_offset)
{
  #pragma omp target teams distribute
  for (int m = 0; m < grid_size; ++m) {
    real s_data_real = 0.0;
    real s_data_imag = 0.0;
    #pragma omp parallel for reduction(+: s_data_real, s_data_imag) 
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
  gpu_find_inner_product_1(
    grid_size, number_of_atoms, real_part, imag_part, other.real_part, other.imag_part, 
    target.real_part, target.imag_part, offset);
#else
  cpu_find_inner_product_1(
    grid_size, number_of_atoms, real_part, imag_part, other.real_part, other.imag_part,
    target.real_part, target.imag_part, offset);
#endif
}

#ifndef CPU_ONLY
void gpu_find_inner_product_2(
  const int number_of_moments,
  const int grid_size,
  const real* __restrict g_inner_product_1_real,
  const real* __restrict g_inner_product_1_imag,
        real* __restrict g_inner_product_2_real,
        real* __restrict g_inner_product_2_imag)
{
  #pragma omp target teams distribute
  for (int m = 0; m < number_of_moments; ++m) {
    real s_data_real = 0.0;
    real s_data_imag = 0.0;
    #pragma omp parallel for reduction(+: s_data_real, s_data_imag) 
    for (int k = 0; k < grid_size; ++k) {
      int n = m * grid_size + k;
      s_data_real += g_inner_product_1_real[n];
      s_data_imag += g_inner_product_1_imag[n];
    }
    g_inner_product_2_real[m] = s_data_real;
    g_inner_product_2_imag[m] = s_data_imag;
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
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE + 1;
#ifndef CPU_ONLY
  gpu_find_inner_product_2(
    number_of_moments, grid_size, real_part, imag_part, target.real_part, target.imag_part);
#else
  cpu_find_inner_product_2(
    number_of_moments, grid_size, real_part, imag_part, target.real_part, target.imag_part);
#endif
}
