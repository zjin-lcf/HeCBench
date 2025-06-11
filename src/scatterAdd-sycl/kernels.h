#define MAX_THREADS_PER_BLOCK 512
#define WORK_SIZE 8192

template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline T atomicAdd(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed, MemoryScope,
                   sycl::access::address_space::generic_space> ref(val);
  return ref.fetch_add(delta);
}

void scatterAdd_kernel(
  const int   *indexes,// [batch_size]
  const float *source, // [batch_size, vector_dim]
  float *outputs,      // [output_size, vector_dim]
  const int batch_size,
  const int output_size,
  const int vector_dim,
  const sycl::nd_item<3> &item,
  float *buffer) 
{
  int warpSize = item.get_sub_group().get_local_range().get(0);
  int thread_idx = item.get_local_id(1) * warpSize  + item.get_local_id(2);
  int batch_idx_start = item.get_group(2) * WORK_SIZE;
  // assert blockDim.x == warpSize
  // assert blockDim.y == MAX_THREADS_PER_BLOCK / warpSize

  float *output_buffer = buffer;
  int *index_buffer = (int*)&buffer[output_size * vector_dim];

  for (int idx_start = 0; idx_start < output_size * vector_dim;
           idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int idx = idx_start + thread_idx;
    if (idx < output_size * vector_dim) {
      output_buffer[idx] = 0;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int idx_start = 0; idx_start < WORK_SIZE; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int batch_idx = batch_idx_start + idx_start + thread_idx;
    if (batch_idx < batch_size) {
      index_buffer[thread_idx] = indexes[batch_idx];
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int buffer_idx_start = 0; buffer_idx_start < MAX_THREADS_PER_BLOCK;
         buffer_idx_start = buffer_idx_start + MAX_THREADS_PER_BLOCK / warpSize) {
      int buffer_idx = buffer_idx_start + item.get_local_id(1);
      int batch_idx = batch_idx_start + idx_start + buffer_idx;
      if (batch_idx < batch_size) {
        int index = index_buffer[buffer_idx];
        for (int j_start = 0; j_start < vector_dim; j_start = j_start + warpSize) {
          int j = j_start + item.get_local_id(2);
          if (j < vector_dim) {
            atomicAdd<float, sycl::memory_scope::work_group>(output_buffer[index * vector_dim + j],
                      source[(size_t)batch_idx * (size_t)vector_dim + (size_t)j]);
          }
        }
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  for (int idx_start = 0; idx_start < output_size * vector_dim;
           idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int idx = idx_start + thread_idx;
    if (idx < output_size * vector_dim) {
      atomicAdd(outputs[idx], output_buffer[idx]);
    }
  }
}

//
// simplify address generations in the kernel
//
void scatterAdd2_kernel(
  const int   *indexes,// [batch_size]
  const float *source, // [batch_size, vector_dim]
  float *outputs,      // [output_size, vector_dim]
  const int batch_size,
  const int output_size,
  const int vector_dim,
  const sycl::nd_item<3> &item,
  float *buffer) 
{
  int warpSize = item.get_sub_group().get_local_range().get(0);
  int thread_idx = item.get_local_id(1) * warpSize  + item.get_local_id(2);
  int batch_idx_start = item.get_group(2) * WORK_SIZE;

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
    item.barrier(sycl::access::fence_space::local_space);

    for (int buffer_idx_start = 0; buffer_idx_start < MAX_THREADS_PER_BLOCK;
         buffer_idx_start = buffer_idx_start + MAX_THREADS_PER_BLOCK / warpSize) {
      int buffer_idx = buffer_idx_start + item.get_local_id(1);
      int batch_idx = batch_idx_start + idx_start + buffer_idx;
      if (batch_idx < batch_size) {
        int index = index_buffer[buffer_idx];
        for (int j = item.get_local_id(2); j < vector_dim; j += warpSize) {
          atomicAdd<float, sycl::memory_scope::work_group>(output_buffer[index * vector_dim + j],
                    source[(size_t)batch_idx * (size_t)vector_dim + (size_t)j]);
        }
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  for (int idx = thread_idx; idx < output_size * vector_dim; idx += MAX_THREADS_PER_BLOCK) {
    atomicAdd(outputs[idx], output_buffer[idx]);
  }
}


