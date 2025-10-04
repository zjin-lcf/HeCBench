void kernel_layerforward_ref(
  const float*__restrict__ input,
        float*__restrict__ input_weights,
        float*__restrict__ hidden_partial_sum,
  const int num_blocks,
  const int hid) 
{
  for (int by = 0; by < num_blocks; by++) {
    float input_node[HEIGHT];
    float weight_matrix[HEIGHT * WIDTH];
    for (int ty = 0; ty < BLOCK_SIZE; ty++)
      for (int tx = 0; tx < BLOCK_SIZE; tx++) {
        int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
        int index_in = HEIGHT * by + ty + 1;
        if ( tx == 0 )
          input_node[ty] = input[index_in] ;
        weight_matrix[ty * WIDTH + tx] = input_weights[index];
      }

    for (int ty = 0; ty < BLOCK_SIZE; ty++)
      for (int tx = 0; tx < BLOCK_SIZE; tx++)
        weight_matrix[ty * WIDTH + tx] = weight_matrix[ty * WIDTH + tx] * input_node[ty];

    for ( int i = 1 ; i <= HEIGHT ; i=i*2){
       int power_two = i; 
       for (int ty = 0; ty < BLOCK_SIZE; ty++)
         for (int tx = 0; tx < BLOCK_SIZE; tx++)
           if( ty % power_two == 0 )
              weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] +
                                              weight_matrix[(ty + power_two/2)* WIDTH + tx];
    }

    for (int ty = 0; ty < BLOCK_SIZE; ty++)
      for (int tx = 0; tx < BLOCK_SIZE; tx++) {
        int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
        input_weights[index] =  weight_matrix[ty * WIDTH + tx];
      }

    for (int ty = 0; ty < BLOCK_SIZE; ty++)
      hidden_partial_sum[by * hid + ty] = weight_matrix[ty];
  }
}

void kernel_adjust_weights_ref (
  const float*__restrict__ ly, 
       float *__restrict__ w, 
  const float*__restrict__ delta, 
        float*__restrict__ oldw, 
  const int num_blocks,
  const int hid) 
{
  for (int by = 0; by < num_blocks; by++) {
    for (int ty = 0; ty < BLOCK_SIZE; ty++)
      for (int tx = 0; tx < BLOCK_SIZE; tx++) {
        int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
        int index_y = HEIGHT * by + ty + 1;
        int index_x = tx + 1;
        w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
        oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
      }
      
    for (int tx = 0; tx < BLOCK_SIZE; tx++) {
      int index_x = tx + 1;
      w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
      oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    }
  }
}


void reference (
  int in, int hid, int out,
  BPNN *net,
  float *input_weights_one_dim,
  const float *input_weights_prev_one_dim,
  float * partial_sum) 
{
  printf("Performing host execution \n");

  float out_err, hid_err;
  unsigned int num_blocks = in / BLOCK_SIZE;

  float* h_input;
  float *h_input_weights;
  float *h_hidden_partial_sum;
  float *h_hidden_delta;
  float *h_input_prev_weights;

  h_input = (float*) malloc(sizeof(float)*(in+1));
  h_input_weights = (float*) malloc(sizeof(float)*(in+1)*(hid+1));
  h_hidden_partial_sum = (float*) malloc(sizeof(float)*num_blocks*WIDTH);

  memcpy(h_input, net->input_units, sizeof(float)*(in+1));
  memcpy(h_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1));

  kernel_layerforward_ref(h_input, h_input_weights, h_hidden_partial_sum, num_blocks, hid);
  memcpy(partial_sum, h_hidden_partial_sum, sizeof(float)*num_blocks*WIDTH);

  for (int j = 1; j <= hid; j++) {
    float sum = 0.f;
    for (unsigned int k = 0; k < num_blocks; k++) {  
      sum += partial_sum[k * hid + j-1] ;
    }
    sum += net->input_weights[0][j];
    net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  // input_weights has been written in the first kernel, so it needs to be restored.
  memcpy(h_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1));
  h_hidden_delta = (float*) malloc(sizeof(float)*(hid+1));
  h_input_prev_weights = (float*) malloc(sizeof(float)*(in+1)*(hid+1));
  memcpy(h_hidden_delta, net->hidden_delta, sizeof(float)*(hid+1));
  memcpy(h_input_prev_weights, input_weights_prev_one_dim, sizeof(float)*(in+1)*(hid+1));
  kernel_adjust_weights_ref(h_input, h_input_weights, h_hidden_delta, h_input_prev_weights, num_blocks, hid);
  memcpy(input_weights_one_dim, h_input_weights, sizeof(float)*(in+1)*(hid+1));

  free(h_input);
  free(h_input_weights);
  free(h_hidden_partial_sum);
  free(h_hidden_delta);
  free(h_input_prev_weights);
}
