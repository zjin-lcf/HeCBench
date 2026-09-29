#include <stdio.h>
#include <stdlib.h>
#include <cctype>
#include <string>
#include <sycl/sycl.hpp>
#include <mpi.h>
#include "../pingpong-cuda/gpu_aware_mpi.h"

void test(sycl::nd_item<1> item, double *d, const long int n) {
  for (long i = item.get_global_id(0);
       i < n; i += item.get_local_range(0) * item.get_group_range(0)) {
    d[i] = d[i] + 1;
  }
}

static int sycl_mpi_gpu_kind(const sycl::device &dev)
{
  std::string vendor = dev.get_info<sycl::info::device::vendor>();
  for (char &c : vendor)
    c = (char)std::tolower((unsigned char)c);
  if (vendor.find("nvidia") != std::string::npos)
    return PINGPONG_GPU_KIND_CUDA;
  if (vendor.find("amd") != std::string::npos ||
      vendor.find("advanced micro") != std::string::npos)
    return PINGPONG_GPU_KIND_HIP;
  if (vendor.find("intel") != std::string::npos)
    return PINGPONG_GPU_KIND_ZE;
  return PINGPONG_GPU_KIND_UNKNOWN;
}

int main(int argc, char *argv[])
{
  /* -------------------------------------------------------------------------------------------
     MPI Initialization
     --------------------------------------------------------------------------------------------*/
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Status stat;

  if(size != 2){
    if(rank == 0){
      printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
    }
    MPI_Finalize();
    exit(0);
  }

  // Map MPI ranks to GPUs
  auto const& gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int num_devices = gpu_devices.size();
  sycl::queue q(gpu_devices[rank % num_devices], sycl::property::queue::in_order());

  pingpong_require_gpu_aware_mpi(sycl_mpi_gpu_kind(q.get_device()), rank);

  //   Loop from 512 KiB to 1 GB (8 * 2^i bytes, i = 16..27)
  for(int i=16; i<=27; i++){

    long int N = 1 << i;

    double *h_A, *d_A;
    h_A = (double*) malloc (N*sizeof(double)); 
    d_A = sycl::malloc_device<double>(N, q);
    q.memset(d_A, 0, N*sizeof(double)).wait();

    const int tag1 = 10;
    const int tag2 = 20;

    int loop_count = 50;

    // Warm-up loop
    for(int i=1; i<=5; i++){
      if(rank == 0){
        MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
        q.wait();
      }
      else if(rank == 1){
        MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
        q.wait();
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for(
            sycl::nd_range<1>(1024*256, 256), [=] (sycl::nd_item<1> item) {
              test(item, d_A, N);
          });
        }).wait();
        MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }
    q.memcpy(h_A, d_A, N*sizeof(double)).wait();
    int valid = 1;
    for (long int j = 0; j < N; j++) {
      if(h_A[j] != 5) {
        fprintf(stderr,
                "ERROR: rank %d: MPI pingpong validation failed at N=%ld index %ld value %.17g\n",
                rank, N, j, h_A[j]);
        fflush(stderr);
        valid = 0;
        break;
      }
    }
    int valid_all = 0;
    MPI_Allreduce(&valid, &valid_all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!valid_all)
      MPI_Abort(MPI_COMM_WORLD, 1);

    free(h_A);

    // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    for(int i=1; i<=loop_count; i++){
      if(rank == 0){
        MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
      }
      else if(rank == 1){
        MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
        MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }

    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    long int num_B = 8*N;
    double num_GB = (double)num_B / 1.0e9;
    double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

    if(rank == 0)
      printf("MPI: Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

    sycl::free(d_A, q);
  }

  MPI_Finalize();

  return 0;
}
