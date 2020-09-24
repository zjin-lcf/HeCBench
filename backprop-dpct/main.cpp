// includes, system
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sys/time.h>

#include "backprop.h"

// cuda kernels
#include "bpnn_layerforward.h"
#include "bpnn_adjust_weights.h"


double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
  int
main( int argc, char** argv) 
{
  setup(argc, argv);
}



int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  float * partial_sum;
  float sum;
  unsigned int num_blocks = in / BLOCK_SIZE;

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
  // todo: fix mem allocation
  int m = 0;
  for (int k = 0; k <= in; k++) {	
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }

  printf("Performing GPU computation\n");

  double offload_start = get_time();

  float* d_input;
 float *d_input_weights;
      float *d_hidden_partial_sum;
     float *d_hidden_delta;
    float *d_input_prev_weights;

  d_input = sycl::malloc_device<float>((in + 1), q_ct1);
  d_input_weights =
      (float *)sycl::malloc_device(sizeof(float) * (in + 1) * (hid + 1), q_ct1);
  d_hidden_partial_sum =
      (float *)sycl::malloc_device(sizeof(float) * num_blocks * WIDTH, q_ct1);

  q_ct1.memcpy(d_input, net->input_units, sizeof(float) * (in + 1));
  q_ct1.memcpy(d_input_weights, input_weights_one_dim,
               sizeof(float) * (in + 1) * (hid + 1));

  sycl::range<3> grid(1, num_blocks, 1);
  sycl::range<3> threads(BLOCK_SIZE, BLOCK_SIZE, 1);
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        input_node_acc_ct1(sycl::range<1>(16 /*HEIGHT*/), cgh);
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        weight_matrix_acc_ct1(sycl::range<1>(256 /*HEIGHT * WIDTH*/), cgh);

    auto dpct_global_range = grid * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_layerforward(d_input, d_input_weights, d_hidden_partial_sum,
                              hid, item_ct1, input_node_acc_ct1.get_pointer(),
                              weight_matrix_acc_ct1.get_pointer());
        });
  });
  q_ct1
      .memcpy(partial_sum, d_hidden_partial_sum,
              sizeof(float) * num_blocks * WIDTH)
      .wait();

    for (int j = 1; j <= hid; j++) {
      sum = 0.f;
      for (unsigned int k = 0; k < num_blocks; k++) {	
        sum += partial_sum[k * hid + j-1] ;
      }
#ifdef DEBUG
      printf("j=%d sum=%f\n", j,sum);
#endif
      sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }

    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

    // input_weights has been written in the first kernel, so it needs to be restored.
  q_ct1.memcpy(d_input_weights, input_weights_one_dim,
               sizeof(float) * (in + 1) * (hid + 1));

  d_hidden_delta = sycl::malloc_device<float>((hid + 1), q_ct1);
  d_input_prev_weights =
      (float *)sycl::malloc_device(sizeof(float) * (in + 1) * (hid + 1), q_ct1);
  q_ct1.memcpy(d_hidden_delta, net->hidden_delta, sizeof(float) * (hid + 1));
  q_ct1.memcpy(d_input_prev_weights, input_weights_prev_one_dim,
               sizeof(float) * (in + 1) * (hid + 1));
  q_ct1.submit([&](sycl::handler &cgh) {
    auto dpct_global_range = grid * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_adjust_weights(d_input, d_input_weights, d_hidden_delta,
                                d_input_prev_weights, hid, item_ct1);
        });
  });
  q_ct1
      .memcpy(input_weights_one_dim, d_input_weights,
              sizeof(float) * (in + 1) * (hid + 1))
      .wait();

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

  sycl::free(d_input, q_ct1);
  sycl::free(d_input_weights, q_ct1);
  sycl::free(d_hidden_partial_sum, q_ct1);
  sycl::free(d_hidden_delta, q_ct1);
  sycl::free(d_input_prev_weights, q_ct1);

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++) 
    printf("i=%d input_units=%f\n", i,net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++) 
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);

  return 0;
}
