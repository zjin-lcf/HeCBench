#define MAX_THREADS_PER_BLOCK 512
#define WORK_SIZE 8192

__global__ void scatterAdd_kernel(
  const int   *indexes,// [batch_size]
  const float *source, // [batch_size, vector_dim]
  float *outputs,      // [output_size, vector_dim]
  const int batch_size,
  const int output_size,
  const int vector_dim
) 
{
  int thread_idx = threadIdx.y * warpSize + threadIdx.x;
  int batch_idx_start = blockIdx.x * WORK_SIZE;
  // assert blockDim.x == warpSize
  // assert blockDim.y == MAX_THREADS_PER_BLOCK / warpSize

  extern __shared__ float buffer[];
  float *output_buffer = buffer;
  int *index_buffer = (int*)&buffer[output_size * vector_dim];

  for (int idx_start = 0; idx_start < output_size * vector_dim; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int idx = idx_start + thread_idx;
    if (idx < output_size * vector_dim) {
      output_buffer[idx] = 0;
    }
  }
  __syncthreads();

  for (int idx_start = 0; idx_start < WORK_SIZE; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int batch_idx = batch_idx_start + idx_start + thread_idx;
    if (batch_idx < batch_size) {
      index_buffer[thread_idx] = indexes[batch_idx];
    }
    __syncthreads();
    for (int buffer_idx_start = 0; buffer_idx_start < MAX_THREADS_PER_BLOCK;
             buffer_idx_start = buffer_idx_start + MAX_THREADS_PER_BLOCK / warpSize) {
      int buffer_idx = buffer_idx_start + threadIdx.y;
      int batch_idx = batch_idx_start + idx_start + buffer_idx;
      if (batch_idx < batch_size) {
        int index = index_buffer[buffer_idx];
        for (int j_start = 0; j_start < vector_dim; j_start = j_start + warpSize) {
          int j = j_start + threadIdx.x;
          if (j < vector_dim) {
            atomicAdd(&output_buffer[index * vector_dim + j], source[(size_t)batch_idx * (size_t)vector_dim + (size_t)j]);
          }
        }
      }
    }
    __syncthreads();
  }

  for (int idx_start = 0; idx_start < output_size * vector_dim; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int idx = idx_start + thread_idx;
    if (idx < output_size * vector_dim) {
      atomicAdd(&outputs[idx], output_buffer[idx]);
    }
  }
}

//
// simplify address generations in the kernel
//
__global__ void scatterAdd2_kernel(
  const int   *indexes,// [batch_size]
  const float *source, // [batch_size, vector_dim]
  float *outputs,      // [output_size, vector_dim]
  const int batch_size,
  const int output_size,
  const int vector_dim
) 
{
  int thread_idx = threadIdx.y * warpSize + threadIdx.x;
  int batch_idx_start = blockIdx.x * WORK_SIZE;

  extern __shared__ float buffer[];
  float *output_buffer = buffer;
  int *index_buffer = (int*)&buffer[output_size * vector_dim];

  for (int idx = thread_idx; idx < output_size * vector_dim; idx += MAX_THREADS_PER_BLOCK) {
    output_buffer[idx] = 0;
  }

  for (int idx_start = 0; idx_start < WORK_SIZE; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int batch_idx = batch_idx_start + idx_start + thread_idx;
    if (batch_idx < batch_size) {
      index_buffer[thread_idx] = indexes[batch_idx];
    }
    __syncthreads();
    for (int buffer_idx_start = 0; buffer_idx_start < MAX_THREADS_PER_BLOCK;
             buffer_idx_start = buffer_idx_start + MAX_THREADS_PER_BLOCK / warpSize) {
      int buffer_idx = buffer_idx_start + threadIdx.y;
      int batch_idx = batch_idx_start + idx_start + buffer_idx;
      if (batch_idx < batch_size) {
        int index = index_buffer[buffer_idx];
        for (int j = threadIdx.x; j < vector_dim; j += warpSize) {
          atomicAdd(&output_buffer[index * vector_dim + j], source[(size_t)batch_idx * (size_t)vector_dim + (size_t)j]);
        }
      }
    }
    __syncthreads();
  }

  for (int idx = thread_idx; idx < output_size * vector_dim; idx += MAX_THREADS_PER_BLOCK) {
    atomicAdd(&outputs[idx], output_buffer[idx]);
  }
}


