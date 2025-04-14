void kernel_layerforward(
  const int numTeams,
  const int numThreads,
  const float*__restrict__ input,
        float*__restrict__ input_weights,
        float*__restrict__ hidden_partial_sum,
  const int hid) 
{
    #pragma omp target teams num_teams(numTeams)
    {
      float input_node[HEIGHT];
      float weight_matrix [HEIGHT * WIDTH];
      #pragma omp parallel num_threads(numThreads)
      {
        int by = omp_get_team_num();
        int tx = omp_get_thread_num() % BLOCK_SIZE;
        int ty = omp_get_thread_num() / BLOCK_SIZE;
        
        int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
        
        int index_in = HEIGHT * by + ty + 1;
        
        if ( tx == 0 )
          input_node[ty] = input[index_in] ;
        #pragma omp barrier
        
        weight_matrix[ty * WIDTH + tx] =  input_weights[index];
        #pragma omp barrier
        
        weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] * input_node[ty];
        #pragma omp barrier
        
        for ( int i = 1 ; i <= HEIGHT ; i=i*2){
          int power_two = i; 
        
          if( ty % power_two == 0 )
          weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two/2)* WIDTH + tx];
        
          #pragma omp barrier
        
        }
        
        input_weights[index] =  weight_matrix[ty * WIDTH + tx];
        
        #pragma omp barrier
        
        if ( tx == 0 ) {
          hidden_partial_sum[by * hid + ty] = weight_matrix[tx* WIDTH + ty];
        }
      }
    }
}

void kernel_adjust_weights (
  const int numTeams,
  const int numThreads,
  const float*__restrict__ ly, 
       float *__restrict__ w, 
  const float*__restrict__ delta, 
        float*__restrict__ oldw, 
  const int hid)
{
    #pragma omp target teams num_teams(numTeams)
    {
      #pragma omp parallel num_threads(numThreads)
      {
        int by = omp_get_team_num();
        int tx = omp_get_thread_num() % BLOCK_SIZE;
        int ty = omp_get_thread_num() / BLOCK_SIZE;
        
        int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
        int index_y = HEIGHT * by + ty + 1;
        int index_x = tx + 1;
        
        w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
        oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
        
        #pragma omp barrier
        
        if (ty == 0 && by ==0){
          w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
          oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
        }
      } 
    }
}
