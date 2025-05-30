#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <mpi.h>

void test(sycl::nd_item<1> item, double *d, const long int n) {
  for (long i = item.get_global_id(0);
       i < n; i += item.get_local_range(0) * item.get_group_range(0)) {
    d[i] = d[i] + 1;
  }
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

  if(size != 2){
    if(rank == 0){
      printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
    }
    MPI_Finalize();
    exit(0);
  }

  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    main_addr = kvs->get_address();
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    kvs = ccl::create_kvs(main_addr);
  }

  //picking a GPU based on localRank, allocate device buffers
  auto const& gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int num_devices = gpu_devices.size();
  sycl::queue q(gpu_devices[rank % num_devices], sycl::property::queue::in_order());

  /* create communicator */
  auto dev = ccl::create_device(q.get_device());
  auto ctx = ccl::create_context(q.get_context());
  auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

  /* create stream */
  auto stream = ccl::create_stream(q);

  //   Loop from 65536 B to 1 GB
  for(int i=16; i<=27; i++){

    long int N = 1 << i;

    double *h_A, *d_A;
    h_A = (double*) malloc (N*sizeof(double)); 
    d_A = sycl::malloc_device<double>(N, q);
    q.memset(d_A, 0, N*sizeof(double)).wait();

    int loop_count = 50;

    // Warm-up and validate NCCL pingpong
    for(int i=1; i<=5; i++){
      if(rank == 0){
        ccl::send(d_A, N, ccl::datatype::float64, 1, comm, stream);
        ccl::recv(d_A, N, ccl::datatype::float64, 1, comm, stream);
      }
      else if(rank == 1){
        ccl::recv(d_A, N, ccl::datatype::float64, 0, comm, stream);
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for(
            sycl::nd_range<1>(1024*256, 256), [=] (sycl::nd_item<1> item) {
              test(item, d_A, N);
          });
        }).wait();
        ccl::send(d_A, N, ccl::datatype::float64, 0, comm, stream);
      }
    }
    q.wait();
    if(rank == 0) {
      q.memcpy(h_A, d_A, N*sizeof(double)).wait();
      for (long int i = 0; i < N; i++) {
        if(h_A[i] != 5) {
          printf("ERROR: CCL pingpong test failed: %lf\n", h_A[i]);
          break;
        }
      }
    }

    free(h_A);

    // Time loop_count iterations of data transfer size 8*N bytes
    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    for(int i=1; i<=loop_count; i++){
      if(rank == 0){
        ccl::send(d_A, N, ccl::datatype::float64, 1, comm, stream);
        ccl::recv(d_A, N, ccl::datatype::float64, 1, comm, stream);
      }
      else if(rank == 1){
        ccl::recv(d_A, N, ccl::datatype::float64, 0, comm, stream);
        ccl::send(d_A, N, ccl::datatype::float64, 0, comm, stream);
      }
    }
    q.wait();

    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    long int num_B = 8*N;
    double num_GB = (double)num_B / 1.0e9;
    double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

    if(rank == 0)
      printf("CCL: Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

    sycl::free(d_A, q);
  }

  MPI_Finalize();

  return 0;
}
