#define MAX_THREADS_PER_BLOCK 512

void scatterAdd_kernel(int *idx, float *src, float *out,
                       int batch_size, int output_size, int vector_dim)
{
  #pragma omp target teams distribute parallel for collapse(2) num_threads(MAX_THREADS_PER_BLOCK)
  for (int d = 0; d < vector_dim; d++) {
    for (int i = 0; i < batch_size; i++) {
      #pragma omp atomic update
      out[idx[i] * vector_dim + d] += src[i * vector_dim + d];
    }
  }
}

void scatterAdd2_kernel(int *idx, float *src, float *out,
                        int batch_size, int output_size, int vector_dim)
{
  #pragma omp target teams distribute parallel for collapse(2) num_threads(MAX_THREADS_PER_BLOCK)
  for (int i = 0; i < batch_size; i++) {
    for (int d = 0; d < vector_dim; d++) {
      #pragma omp atomic update
      out[idx[i] * vector_dim + d] += src[i * vector_dim + d];
    }
  }
}
