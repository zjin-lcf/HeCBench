#pragma omp target teams num_teams((NUM*NUM/2)/BLOCK_SIZE) thread_limit(BLOCK_SIZE)
{
  Real sum_cache[BLOCK_SIZE];
  #pragma omp parallel
  {
    int lid = omp_get_thread_num();
    int tid = omp_get_team_num();
    int gid = tid * BLOCK_SIZE + lid;
    int row = (gid % (NUM/2)) + 1; 
    int col = (gid / (NUM/2)) + 1; 

    int NUM_2 = NUM >> 1;

    Real pres_r = pres_red(col, row);
    Real pres_b = pres_black(col, row);

    // add squared pressure
    sum_cache[lid] = (pres_r * pres_r) + (pres_b * pres_b);

    // synchronize threads in block to ensure all thread values stored
    #pragma omp barrier

    // add up values for block
    int i = BLOCK_SIZE >> 1;
    while (i != 0) {
      if (lid < i) {
        sum_cache[lid] += sum_cache[lid + i];
      }
      #pragma omp barrier
      i >>= 1;
    }

    // store block's summed values
    if (lid == 0) {
      pres_sum[tid] = sum_cache[0];
    }
  }
}
