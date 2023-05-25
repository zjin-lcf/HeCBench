#include "track_ellipse.h"
#include <sycl/sycl.hpp>

// Host and device arrays to hold matrices for all cells
// (so we can copy to and from the device in a single transfer)

// The number of work items per work group
#define LOCAL_WORK_SIZE 256
#define FP_TYPE float
#define FP_CONST(num) num##f
#define PI_FP32 FP_CONST(3.14159)
#define ONE_OVER_PI (FP_CONST(1.0) / PI_FP32)
#define MU FP_CONST(0.5)
#define LAMBDA (FP_CONST(8.0) * MU + FP_CONST(1.0))
#define NEXT_LOWEST_POWER_OF_TWO 256

//---------------  device function ---------------------------
FP_TYPE heaviside(FP_TYPE x) {
  return (sycl::atan(x) * ONE_OVER_PI) + FP_CONST(0.5);
}

// Host function that launches an OpenCL kernel to compute the MGVF matrices for the specified cells
void IMGVF_SYCL(sycl::queue &q, MAT **IE, MAT **IMGVF,
		double vx, double vy, double e, int max_iterations, double cutoff, int num_cells) {

  // Initialize the data on the GPU
  // Allocate array of offsets to each cell's image
  size_t mem_size = sizeof(int) * num_cells;
  int* host_I_offsets = (int *) malloc(mem_size);

  // Allocate arrays to hold the dimensions of each cell's image
  int* host_m_array = (int *) malloc(mem_size);
  int* host_n_array = (int *) malloc(mem_size);

  // Figure out the size of all of the matrices combined
  int i, j;
  size_t total_size = 0;
  for (int cell_num = 0; cell_num < num_cells; cell_num++) {
    MAT *I = IE[cell_num];
    size_t size = I->m * I->n;
    total_size += size;
  }
  size_t total_mem_size = total_size * sizeof(float);

  // Allocate host memory just once for all cells
  float* host_I_all = (float *) malloc(total_mem_size);

  // Copy each initial matrix into the allocated host memory
  int offset = 0;
  for (int cell_num = 0; cell_num < num_cells; cell_num++) {
    MAT *I = IE[cell_num];

    // Determine the size of the matrix
    int m = I->m, n = I->n;
    int size = m * n;

    // Store memory dimensions
    host_m_array[cell_num] = m;
    host_n_array[cell_num] = n;

    // Store offsets to this cell's image
    host_I_offsets[cell_num] = offset;

    // Copy matrix I (which is also the initial IMGVF matrix) into the overall array
    for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
        host_I_all[offset + (i * n) + j] = (float) m_get_val(I, i, j);

    offset += size;
  }

  // Setup execution parameters
  size_t num_work_groups = num_cells;
  size_t gws = num_work_groups * LOCAL_WORK_SIZE;

  // Convert double-precision parameters to single-precision
  float vx_float = (float) vx;
  float vy_float = (float) vy;
  //float e_float = (float) e;
  float cutoff_float = (float) cutoff;

  int *d_I_offsets = sycl::malloc_device<int>(num_cells, q);
  q.memcpy(d_I_offsets, host_I_offsets, sizeof(int)*num_cells);

  int *d_m_array = sycl::malloc_device<int>(num_cells, q);
  q.memcpy(d_m_array, host_m_array, sizeof(int)*num_cells);

  int *d_n_array = sycl::malloc_device<int>(num_cells, q);
  q.memcpy(d_n_array, host_n_array, sizeof(int)*num_cells);

  float *d_I_all = sycl::malloc_device<float>(total_size, q);
  q.memcpy(d_I_all, host_I_all, sizeof(float)*total_size);

  float *d_IMGVF_all = sycl::malloc_device<float>(total_size, q);
  q.memcpy(d_IMGVF_all, host_I_all, sizeof(float)*total_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor <float, 1> IMGVF_acc(sycl::range<1>(41 * 81), cgh);
    sycl::local_accessor <float, 1> IMGVF_buffer_acc(sycl::range<1>(LOCAL_WORK_SIZE), cgh);
    sycl::local_accessor <int, 1> cell_converged_acc(1, cgh);
    // Compute the MGVF on the GPU
    cgh.parallel_for<class IMGVF>(
        sycl::nd_range<1>(sycl::range<1>(gws), sycl::range<1>(LOCAL_WORK_SIZE)),
        [=] (sycl::nd_item<1> item) {
        #include "kernel_IMGVF.sycl"
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (IMGVF): %f (s)\n", time * 1e-9f);

  q.memcpy(host_I_all, d_IMGVF_all, sizeof(float)*total_size).wait();

  sycl::free(d_I_offsets, q);
  sycl::free(d_m_array, q);
  sycl::free(d_n_array, q);
  sycl::free(d_I_all, q);
  sycl::free(d_IMGVF_all, q);

  // Copy each result matrix into its appropriate host matrix
  offset = 0;
  for (int cell_num = 0; cell_num < num_cells; cell_num++) {
    MAT *IMGVF_out = IMGVF[cell_num];

    // Determine the size of the matrix
    int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
    // Pack the result into the matrix
    for (i = 0; i < m; i++)
      for (j = 0; j < n; j++) {
#ifdef DEBUG
        printf("host_IMGVF: %f\n",host_I_all[offset + (i * n) + j]);
#endif

        m_set_val(IMGVF_out, i, j, (double) host_I_all[offset + (i * n) + j]);
      }

    offset += (m * n);
  }

  // Free host memory
  free(host_m_array);
  free(host_n_array);
  free(host_I_all);
  free(host_I_offsets);
}
