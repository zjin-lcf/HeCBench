#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "lud.h"

#define BLOCK_SIZE 16

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

int main ( int argc, char *argv[] )
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:",
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'i':
        input_file = optarg;
        break;
      case 'v':
        do_verify = 1;
        break;
      case 's':
        matrix_dim = atoi(optarg);
        printf("Generate input matrix internally, size =%d\n", matrix_dim);
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
            argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  }

  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }

  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  /* beginning of timing point */
  stopwatch_start(&sw);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_m = sycl::malloc_device<float>(matrix_dim * matrix_dim, q);
  q.memcpy(d_m, m, sizeof(float) * matrix_dim * matrix_dim);

  sycl::range<1> global_work1(BLOCK_SIZE);
  sycl::range<1> local_work1(BLOCK_SIZE);
  int offset;
  int i=0;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {

    offset = i;  // add the offset

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor <float, 1> shadow (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
      cgh.parallel_for<class diagonal>(
        sycl::nd_range<1>(global_work1, local_work1), [=] (sycl::nd_item<1> item) {
        #include "kernel_lud_diagonal.sycl"
      });
    });

    sycl::range<1> global_work2 (BLOCK_SIZE * 2 * ((matrix_dim-i)/BLOCK_SIZE-1));
    sycl::range<1> local_work2 (BLOCK_SIZE * 2);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor <float, 1> dia (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
      sycl::local_accessor <float, 1> peri_row (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
      sycl::local_accessor <float, 1> peri_col (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
      cgh.parallel_for<class peri>(
        sycl::nd_range<1>(global_work2, local_work2), [=] (sycl::nd_item<1> item) {
        #include "kernel_lud_perimeter.sycl"
      });
    });

    sycl::range<2> global_work3(BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1), BLOCK_SIZE * ((matrix_dim-i)/BLOCK_SIZE-1));
    sycl::range<2> local_work3(BLOCK_SIZE, BLOCK_SIZE);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor <float, 1> peri_row (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
      sycl::local_accessor <float, 1> peri_col (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
      cgh.parallel_for<class internal>(
        sycl::nd_range<2>(global_work3, local_work3) , [=] (sycl::nd_item<2> item) {
        #include "kernel_lud_internal.sycl"
      });
    });
  } // for

  offset = i;  // add the offset

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor <float, 1> shadow (sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
    cgh.parallel_for<class diagonal2>(
      sycl::nd_range<1>(global_work1, local_work1), [=] (sycl::nd_item<1> item) {
      #include "kernel_lud_diagonal.sycl"
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time : %f (s)\n", time * 1e-9f);

  q.memcpy(m, d_m, sizeof(float) * matrix_dim * matrix_dim).wait();

  /* end of timing point */
  stopwatch_stop(&sw);
  printf("Device offloading time (s): %lf\n", get_interval_by_sec(&sw));

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim);
    free(mm);
  }

  free(m);
  sycl::free(d_m, q);
}
