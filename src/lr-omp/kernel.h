// begin of linear_regression
void linear_regression(
  const int numTeams,
  const int numThreads,
  const float2 *__restrict dataset,
        float4 *__restrict result)
{
  #pragma omp target teams distribute num_teams(numTeams) 
  for ( size_t block_id = 0 ; block_id < numTeams ; ++ block_id ) {
    float sum_x = 0, sum_y = 0, sum_z = 0, sum_w = 0;
    #pragma omp parallel for reduction (+ : sum_x, sum_y, sum_z, sum_w) num_threads(numThreads)
    for ( size_t thread_id = 0 ; thread_id < numThreads ; ++ thread_id ) {
      size_t glob_id = block_id * numThreads + thread_id ;
      float x = dataset [ glob_id ].x ;
      float y = dataset [ glob_id ].y ;
      sum_x += x ;
      sum_y += y ;
      sum_z += x * y ;
      sum_w += x * x ;
    }
    result [ block_id ] = {sum_x, sum_y, sum_z, sum_w}; 
  }
}
// end of linear_regression

// begin of rsquared
void rsquared(
  const int numTeams,
  const int numThreads,
  const float2 *__restrict dataset,
  const float mean,
  const float2 equation, // [a0,a1]
  float2 *__restrict result)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for ( size_t block_id = 0 ; block_id < numTeams ; ++ block_id ) {
    float sum_x = 0, sum_y = 0;
    #pragma omp parallel for reduction (+ : sum_x, sum_y) num_threads(numThreads)
    for ( size_t thread_id = 0 ; thread_id < numThreads ; ++ thread_id ) {
      size_t glob_id = block_id * numThreads + thread_id ;
      sum_x += powf((dataset[glob_id].y - mean), 2.f);
      float y_estimated = dataset[glob_id].x * equation.y + equation.x;
      sum_y += powf((y_estimated - mean), 2.f);
    }
    result [ block_id ] = {sum_x, sum_y};
  }
}
// end of rsquared
