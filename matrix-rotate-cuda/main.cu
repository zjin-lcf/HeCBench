#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>

__global__ void rotate_matrix_parallel (float *matrix, const int n) {
  int layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer < n/2) {
    int first = layer;
    int last = n - 1 - layer;
    for(int i = first; i < last; ++i) {
      int offset = i - first;

      float top = matrix[first*n+i]; // save top
      // left -> top
      matrix[first*n+i] = matrix[(last-offset)*n+first];

      // bottom -> left
      matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];

      // right -> bottom
      matrix[last*n+(last-offset)] = matrix[i*n+last];

      // top -> right
      matrix[i*n+last] = top; // right <- saved top
    }
  }
}

void rotate_matrix_serial(float *matrix, int n) {

  for (int layer = 0; layer < n / 2; ++layer) {
    int first = layer;
    int last = n - 1 - layer;
    for(int i = first; i < last; ++i) {
      int offset = i - first;
        float top = matrix[first*n+i]; // save top
        // left -> top
        matrix[first*n+i] = matrix[(last-offset)*n+first];

        // bottom -> left
        matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];

        // right -> bottom
        matrix[last*n+(last-offset)] = matrix[i*n+last];

        // top -> right
        matrix[i*n+last] = top; // right <- saved top
    }
  }
}

int main(int argc, char** argv) {

  const int n = atoi(argv[1]);
  float *serial_res = (float*) aligned_alloc(1024, n*n*sizeof(float));
  float *parallel_res = (float*) aligned_alloc(1024, n*n*sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      serial_res[i*n+j] = parallel_res[i*n+j] = i*n+j;

  float *d_parallel_res;
  cudaMalloc((void**)&d_parallel_res, n*n*sizeof(float));
  cudaMemcpy(d_parallel_res, parallel_res, n*n*sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < 100; i++) {
    rotate_matrix_serial(serial_res, n);
    rotate_matrix_parallel<<<(n/2+255)/256, 256>>>(d_parallel_res, n);
  }
  cudaMemcpy(parallel_res, d_parallel_res, n*n*sizeof(float), cudaMemcpyDeviceToHost);

  int errors = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (serial_res[i*n+j] != parallel_res[i*n+j]) {
        errors++; 
        break;
      }
    }
  }
  if (errors) 
    printf("fail\n");
  else 
    printf("success\n");

  free(serial_res);
  free(parallel_res);
  cudaFree(d_parallel_res);
  return 0;
}

