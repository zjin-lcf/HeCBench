// Copyright (c) 2015 Matthias Noack (ma.noack.pr@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utils.hpp"
#include "kernels.cu"

long benchmark(
    complex_t *sigma_in,
    complex_t *sigma_out,
    complex_t *hamiltonian,
    size_t size_sigma,
    size_t size_hamiltonian,
    complex_t *sigma_reference,
    complex_t *sigma_reference_transformed ,
    const int dim, 
    const int num, // global_work_size
    const int kernel_id, 
    size_t vec_length, 
    decltype(&transform_matrices_aos_to_aosoa) transformation_sigma,
    bool scale_hamiltonian,
    decltype(&transform_matrix_aos_to_soa) transformation_hamiltonian)
{
  initialise_hamiltonian(hamiltonian, dim);

  if (scale_hamiltonian) 
    transform_matrix_scale_aos(hamiltonian, dim); // pre-scale hamiltonian

  if (transformation_hamiltonian)
    transformation_hamiltonian(hamiltonian, dim);  

  initialise_sigma(sigma_in, sigma_out, dim, num);

  std::memcpy(sigma_reference_transformed, sigma_reference, size_sigma * sizeof(complex_t));

  // transform memory layout if a transformation is specified
  if (transformation_sigma) {
    // transform reference for comparison
    transformation_sigma(sigma_reference_transformed, dim, num, vec_length);

    // transform sigma
    transformation_sigma(sigma_in, dim, num, vec_length);
  }

  // extract the real and imag data 
  real_2_t* ham = allocate_aligned<real_2_t>(size_hamiltonian);
  real_2_t* sin = allocate_aligned<real_2_t>(size_sigma);
  real_2_t* sout = allocate_aligned<real_2_t>(size_sigma);
  
  for (size_t i = 0; i < size_hamiltonian; i++) {
    ham[i].x = hamiltonian[i].real(); 
    ham[i].y = hamiltonian[i].imag(); 
  }

  for (size_t i = 0; i < size_sigma; i++) {
    sin[i].x = sigma_in[i].real(); 
    sin[i].y = sigma_in[i].imag(); 
  }

  for (size_t i = 0; i < size_sigma; i++) {
    sout[i].x = sigma_out[i].real(); 
    sout[i].y = sigma_out[i].imag(); 
  }

  // allocate device memory
  real_2_t *d_hamiltonian;
  real_2_t *d_sigma_in;
  real_2_t *d_sigma_out;

  hipMalloc((void**)&d_hamiltonian, sizeof(real_2_t) * size_hamiltonian);
  hipMemcpy(d_hamiltonian, ham, sizeof(real_2_t) * size_hamiltonian,
             hipMemcpyHostToDevice);

  hipMalloc((void**)&d_sigma_in, sizeof(real_2_t) * size_sigma);
  hipMemcpy(d_sigma_in, sin, sizeof(real_2_t) * size_sigma,
             hipMemcpyHostToDevice);

  hipMalloc((void**)&d_sigma_out, sizeof(real_2_t) * size_sigma);

  long total_time = 0;

  // benchmark loop
  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {

    // clear output 
    hipMemcpy(d_sigma_out, sout, sizeof(real_2_t) * size_sigma,
               hipMemcpyHostToDevice);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // empty kernel
    switch(kernel_id) {
      case 0:  {
        dim3 k0_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k0_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_empty, dim3(k0_gws), dim3(k0_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // initial kernel
      case 1: {
        dim3 k1_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k1_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_init, dim3(k1_gws), dim3(k1_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // refactored initial kernel
      case 2: {
        dim3 k2_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k2_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_refactor, dim3(k2_gws), dim3(k2_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // refactored initial kernel with direct store
      case 3: {
        dim3 k3_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k3_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_refactor_direct_store, dim3(k3_gws), dim3(k3_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range
      case 4: {
        dim3 k4_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k4_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_naive, dim3(k4_gws), dim3(k4_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range and compile time constants
      case 5: {
        dim3 k5_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k5_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_naive_constants, dim3(k5_gws), dim3(k5_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range, compile time constants, and permuted loops with temporaries
      case 6: {
        dim3 k6_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k6_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_naive_constants_perm, dim3(k6_gws), dim3(k6_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 1D range and direct store
      case 7: {
        dim3 k7_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k7_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_naive_direct, dim3(k7_gws), dim3(k7_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range, compile time constants, and direct store
      case 8: {
        dim3 k8_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k8_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_naive_constants_direct, dim3(k8_gws), dim3(k8_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 1D range, compile time constants, direct store, and permuted loops with temporaries
      case 9: {
        dim3 k9_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k9_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_naive_constants_direct_perm, dim3(k9_gws), dim3(k9_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 2D-range
      case 10: {
        dim3 k10_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k10_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa, dim3(k10_gws), dim3(k10_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 2D-range and compile-time constants
      case 11: {
        dim3 k11_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k11_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_constants, dim3(k11_gws), dim3(k11_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 2D-range, compile-time constants, and permuted loops with temporaries
      case 12: {
        dim3 k12_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k12_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_constants_perm, dim3(k12_gws), dim3(k12_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 2D range and direct store
      case 13: {
        dim3 k13_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k13_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_direct, dim3(k13_gws), dim3(k13_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 2D range, compile-time constants, and direct store
      case 14: {
        dim3 k14_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k14_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_constants_direct, dim3(k14_gws), dim3(k14_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with compile-time constants, direct store, and permuted loops with temporaries
      case 15: {
        dim3 k15_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k15_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        hipLaunchKernelGGL(comm_aosoa_constants_direct_perm, dim3(k15_gws), dim3(k15_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel
      case 16: {
        dim3 k16_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k16_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa, dim3(k16_gws), dim3(k16_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // manually vectorised kernel with compile-time constants
      case 17: {
        dim3 k17_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k17_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa_constants, dim3(k17_gws), dim3(k17_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with compile-time constants and permuted loops with temporaries
      case 18: {
        dim3 k18_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k18_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa_constants_perm, dim3(k18_gws), dim3(k18_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with compile-time constants and prefetch
      case 19: {
        dim3 k19_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k19_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa_constants_perm_prefetch, dim3(k19_gws), dim3(k19_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with direct store
      case 20: {
        dim3 k20_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k20_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa_direct, dim3(k20_gws), dim3(k20_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // manually vectorised kernel with compile time constants and direct store
      case 21: {
        dim3 k21_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k21_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa_constants_direct, dim3(k21_gws), dim3(k21_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with compile time constants, direct store, and prefetch
      case 22: {
        dim3 k22_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k22_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa_constants_direct_prefetch, dim3(k22_gws), dim3(k22_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

       // manually vectorised kernel with compile time constants, direct store, and permuted loops with temporaries
      case 23: {
        dim3 k23_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k23_lws (VEC_LENGTH);
        hipLaunchKernelGGL(comm_manual_aosoa_constants_direct_perm, dim3(k23_gws), dim3(k23_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // final GPGPU kernel optimised for an Nvidia GPU
      case 24: {
        size_t block_dim_x = (dim * dim + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        size_t block_dim_y = NUM_SUB_GROUPS;
        dim3 k24_gws (num / (block_dim_y * CHUNK_SIZE), 1);
        dim3 k24_lws (block_dim_x, block_dim_y);
        hipLaunchKernelGGL(final_gpu_kernel, dim3(k24_gws), dim3(k24_lws), 0, 0, d_sigma_in, d_sigma_out, d_hamiltonian, num);
        break;
      }
      default: std::cerr << "ERROR: **** benchmark kernel unavailable **** \n";
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  std::cout << "Total execution time of kernel "
            << look_up(kernel_id)  << " : " << total_time * 1e-9 << " (s)" << std::endl;

  real_t deviation = 0;

  if (kernel_id > 0)  {

    hipMemcpy(sout, d_sigma_out, sizeof(real_2_t) * size_sigma, hipMemcpyDeviceToHost);

    for (size_t i = 0; i < size_sigma; i++) {
      sigma_out[i] = {sout[i].x, sout[i].y};
    }

    // measure the differences between the CPU and GPU results 
    deviation = compare_matrices(sigma_out, sigma_reference_transformed, dim, num);

    std::cout << "Deviation of kernel " << look_up(kernel_id) << ": " << deviation;
  } else {
    // the deviation of an empty kernel does not make sense
    std::cout << "Deviation of kernel " << look_up(kernel_id) << "N/A";
  }

  std::cout << std::endl << std::endl;

  hipFree(d_hamiltonian);
  hipFree(d_sigma_in);
  hipFree(d_sigma_out);
  free(sin);
  free(sout);
  free(ham);

  return total_time;
}

int main(int argc, char* argv[])
{
  // debugging
  print_compile_config(std::cout);

  // constants
  const size_t dim = DIM;
  const size_t num = NUM;

  // allocate host memory
  size_t size_hamiltonian = dim * dim;
  size_t size_sigma = size_hamiltonian * num;
  size_t size_sigma_byte = sizeof(complex_t) * size_sigma;

  complex_t* hamiltonian = allocate_aligned<complex_t>(size_hamiltonian);
  complex_t* sigma_in = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_out = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_reference = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_reference_transformed = allocate_aligned<complex_t>(size_sigma);

  // perform reference computation for correctness analysis
  initialise_hamiltonian(hamiltonian, dim);
  initialise_sigma(sigma_in, sigma_out, dim, num);
  commutator_reference(sigma_in, sigma_out, hamiltonian, dim, num);

  // copy reference results
  std::memcpy(sigma_reference, sigma_out, size_sigma_byte);

  // total kernel time for all benchmarks 
  long ktime = 0;

  // The macro "BENCHMARK(...)" is defined in utils.hpp
  ktime += BENCHMARK(0, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(1, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(2, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(3, VEC_LENGTH, NO_TRANSFORM, SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(4, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(5, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(6, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(7, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(8, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(9, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(10, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(11, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(12, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(13, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(14, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(15, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(16, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(17, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(18, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(19, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(20, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(21, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(22, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(23, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(24, 2, NO_TRANSFORM, SCALE_HAMILT, NO_TRANSFORM);

  printf("Total kernel time for all benchmarks %lf (s)\n", ktime * 1e-9);

  free(hamiltonian);
  free(sigma_in);
  free(sigma_out);
  free(sigma_reference);
  free(sigma_reference_transformed);

  return 0;
}
