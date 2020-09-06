#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include <sys/time.h>
#include <getopt.h>
#include "common.h"

#define BLOCK_SIZE 16

#include "lud_kernels.cpp"

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



  int
main ( int argc, char *argv[] )
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
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
        // fprintf(stderr, "Currently not supported, use -i instead\n");
        // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
        // exit(EXIT_FAILURE);
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

  float *d_m;
    d_m = sycl::malloc_device<float>(matrix_dim * matrix_dim, q_ct1);
    q_ct1.memcpy(d_m, m, matrix_dim * matrix_dim * sizeof(float)).wait();

  int offset;
  int i=0;
  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
    offset = i;  // add the offset
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<float, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                shadow_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*BLOCK_SIZE*/),
                               cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCK_SIZE),
                                  sycl::range<3>(1, 1, BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                    lud_diagonal(d_m, matrix_dim, offset, item_ct1,
                                 shadow_acc_ct1.get_pointer());
                });
        });
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<float, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                dia_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*BLOCK_SIZE*/), cgh);
            sycl::accessor<float, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                peri_row_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*BLOCK_SIZE*/),
                                 cgh);
            sycl::accessor<float, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                peri_col_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*BLOCK_SIZE*/),
                                 cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, (matrix_dim - i) / BLOCK_SIZE - 1) *
                        sycl::range<3>(1, 1, 2 * BLOCK_SIZE),
                    sycl::range<3>(1, 1, 2 * BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                    lud_perimeter(d_m, matrix_dim, offset, item_ct1,
                                  dia_acc_ct1.get_pointer(),
                                  peri_row_acc_ct1.get_pointer(),
                                  peri_col_acc_ct1.get_pointer());
                });
        });
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<float, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                peri_row_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*BLOCK_SIZE*/),
                                 cgh);
            sycl::accessor<float, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                peri_col_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*BLOCK_SIZE*/),
                                 cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, (matrix_dim - i) / BLOCK_SIZE - 1,
                                   (matrix_dim - i) / BLOCK_SIZE - 1) *
                        sycl::range<3>(1, BLOCK_SIZE, BLOCK_SIZE),
                    sycl::range<3>(1, BLOCK_SIZE, BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                    lud_internal(d_m, matrix_dim, offset, item_ct1,
                                 peri_row_acc_ct1.get_pointer(),
                                 peri_col_acc_ct1.get_pointer());
                });
        });
  } // for

  offset = i;  // add the offset
    q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            shadow_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*BLOCK_SIZE*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCK_SIZE),
                                           sycl::range<3>(1, 1, BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             lud_diagonal(d_m, matrix_dim, offset, item_ct1,
                                          shadow_acc_ct1.get_pointer());
                         });
    });

    q_ct1.memcpy(m, d_m, matrix_dim * matrix_dim * sizeof(float)).wait();

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
    sycl::free(d_m, q_ct1);
  return 0;
}        
