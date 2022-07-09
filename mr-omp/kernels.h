void mr32_sf(
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int j = 0; j < iter; j++) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      #pragma omp atomic update
      *val += straightforward_mr32(bases, cnt, n);
    }
  }
}

void mr32_eff(
  const uint32_t *__restrict bases,
  const uint32_t *__restrict n32,
  int *__restrict val,
  int iter)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int j = 0; j < iter; j++) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      #pragma omp atomic update
      *val += efficient_mr32(bases, cnt, n);
    }
  }
}
