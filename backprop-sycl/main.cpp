#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "common.h"
#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////


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
int main( int argc, char** argv) 
{
  setup(argc, argv);
  return 0;
}

int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  float *d_input = malloc_device<float>(in+1, q);
  q.memcpy(d_input, net->input_units, sizeof(float)*(in+1));

  float *d_input_weights = malloc_device<float>((in+1)*(hid+1), q);
  q.memcpy(d_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1));

  float *d_hidden_partial_sum = malloc_device<float>(num_blocks*WIDTH, q);

  // set global and local workitems
  range<2> global_work(BLOCK_SIZE*num_blocks, BLOCK_SIZE);
  range<2> local_work(BLOCK_SIZE, BLOCK_SIZE);

  q.submit([&](handler& cgh) {
    accessor <float, 1, sycl_read_write, access::target::local> input_node (HEIGHT, cgh);
    accessor <float, 1, sycl_read_write, access::target::local> weight_matrix (HEIGHT * WIDTH, cgh);
    cgh.parallel_for<class forward>(nd_range<2>(global_work, local_work), [=] (nd_item<2> item) {
      #include "bpnn_layerforward.sycl"
    });
  });

  q.memcpy(partial_sum, d_hidden_partial_sum, sizeof(float)*num_blocks*WIDTH).wait();

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {  
      sum += partial_sum[k * hid + j-1] ;
    }
#ifdef DEBUG
      printf("j=%d sum=%f\n", j,sum);
#endif
    sum += net->input_weights[0][j];
    net-> hidden_units[j] = float(1.0 / (1.0 + std::exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  // input_weights_sycl has been written in the first kernel, so it needs to be restored.
  q.memcpy(d_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1));

  float *d_hidden_delta = malloc_device<float>(hid+1, q);
  q.memcpy(d_hidden_delta, net->hidden_delta, sizeof(float)*(hid+1));

  float *d_input_prev_weights = malloc_device<float>((in+1)*(hid+1), q);
  q.memcpy(d_input_prev_weights, input_weights_prev_one_dim, sizeof(float)*(in+1)*(hid+1));

  q.submit([&](handler& cgh) {
    cgh.parallel_for<class adjust_weights>(nd_range<2>(global_work, local_work), [=] (nd_item<2> item) {
      #include "bpnn_adjust_weights.sycl"
    });
  });

  q.memcpy(input_weights_one_dim, d_input_weights, sizeof(float)*(in+1)*(hid+1)).wait();

  free(d_input, q);
  free(d_input_weights, q);
  free(d_hidden_partial_sum, q);
  free(d_hidden_delta, q);
  free(d_input_prev_weights, q);

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

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
