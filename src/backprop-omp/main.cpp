// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "backprop.h"
#include "bpnn_kernels.h"

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

  printf("Performing device offload\n");

  double offload_start = get_time();

  float* input = net->input_units;
  float* input_weights = input_weights_one_dim;
  float* input_prev_weights = input_weights_prev_one_dim;
  float* hidden_delta = net->hidden_delta;

  const int numTeams = num_blocks; 
  const int numThreads = BLOCK_SIZE * BLOCK_SIZE;

  #pragma omp target data map(to: input[0:in+1], \
                                  hidden_delta[0:hid+1], \
                                  input_prev_weights[0:(in+1)*(hid+1)]) \
                          map(tofrom: input_weights[0:(in+1)*(hid+1)]) \
                          map(alloc: partial_sum[0:num_blocks*WIDTH]) 
  {
    kernel_layerforward(numTeams, numThreads, input, input_weights, partial_sum, hid);
    
    #pragma omp target update from(partial_sum[0:num_blocks*WIDTH])
    for (int j = 1; j <= hid; j++) {
      sum = 0.0;
      for (int k = 0; k < num_blocks; k++) {  
        sum += partial_sum[k * hid + j-1] ;
      }
    #ifdef DEBUG
      printf("j=%d sum=%f\n", j,sum);
    #endif
      sum += net->input_weights[0][j];
      net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }
    
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
    
    // input_weights has been written in the first kernel, so it needs to be restored.
    #pragma omp target update to(input_weights[0:(in+1)*(hid+1)])
    
    kernel_adjust_weights(numTeams, numThreads, input, input_weights, hidden_delta, input_prev_weights, hid);
  }
  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++) 
    printf("i=%d input_units=%f\n", i, net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++) 
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);

  return 0;
}
