void map_ref(const int *__restrict__ pages,
             const float *__restrict__ page_ranks,
                   float *__restrict__ maps,
             const unsigned int *__restrict__ noutlinks,
             const int n)
{
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    float outbound_rank = page_ranks[i]/(float)noutlinks[i];
    for(int j=0; j<n; ++j){
      maps[(size_t)i*n+j] = pages[(size_t)i*n+j]*outbound_rank;
    }
  }
}

void reduce_ref(float *__restrict__ page_ranks,
                const float *__restrict__ maps,
                const int n,
                float *__restrict__ dif)
{
  const float D_FACTOR = 0.85f;
  #pragma omp parallel for
  for (int j = 0; j < n; j++) {
    float old_rank = page_ranks[j];
    float new_rank = 0.0f;
    for(int i=0; i< n; ++i){
      new_rank += maps[(size_t)i*n + j];
    }

    new_rank = ((1.f-D_FACTOR)/n)+(D_FACTOR*new_rank);
    dif[j] = fmaxf(fabsf(new_rank - old_rank), dif[j]);
    page_ranks[j] = new_rank;
  }
}
