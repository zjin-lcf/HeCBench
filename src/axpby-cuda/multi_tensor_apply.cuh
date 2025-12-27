// This header is the one-stop shop for all your multi-tensor apply needs.

// TODO:  Kernel arg size limit may be <4KB for some other cards (ie Jetson)
constexpr int depth_to_max_tensors[6] = {110, 64, 48, 36, 30, 24};
constexpr int depth_to_max_blocks[6] = {320, 320, 320, 320, 320, 320};

template <typename T>
struct Tensor {
  T *data_ptr;
  int64_t numel; 
};

template <int n>
struct TensorListMetadata {
  void*   addresses[n][depth_to_max_tensors[n - 1]];
  int64_t        sizes[depth_to_max_tensors[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int            block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(int64_t chunk_size, volatile int* noop_flag, T tl, U callable,
                                          ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(chunk_size, noop_flag, tl, args...);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(int64_t block_size, int64_t chunk_size, const Tensor<int>& noop_flag,
                        const std::vector<std::vector<Tensor<float>>>& tensor_lists, T callable, ArgTypes... args) {

  int ntensors = tensor_lists[0].size();

  TensorListMetadata<depth> tl;

  tl.start_tensor_this_launch = 0;
  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (int t = 0; t < ntensors; t++) {
    tl.sizes[loc_tensor_info] = tensor_lists[0][t].numel;
    for (int d = 0; d < depth; d++) tl.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr;
    loc_tensor_info++;

    auto chunks_this_tensor = (tensor_lists[0][t].numel + chunk_size - 1) / chunk_size;

    for (auto chunk = 0; chunk < chunks_this_tensor; chunk++) {
      // std::cout << chunks_this_tensor << std::endl;
      tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tl.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth - 1] && chunk == chunks_this_tensor - 1);
      bool blocks_full = (loc_block_info == depth_to_max_blocks[depth - 1]);
      bool last_chunk = (t == ntensors - 1 && chunk == chunks_this_tensor - 1);
      if (tensors_full || blocks_full || last_chunk) {
        // using accscalar_t = acc_type<scalar_t, true>;
        multi_tensor_apply_kernel<<<loc_block_info, block_size>>>(chunk_size, noop_flag.data_ptr, tl,
                                                                  callable, args...);

        // Reset.
        loc_block_info = 0;
        if (chunk == chunks_this_tensor - 1) {
          // std::cout << "Hit case 1 " << cond1 << " " << cond2 << " " << cond3 << std::endl;
          loc_tensor_info = 0;
          tl.start_tensor_this_launch = t + 1;
        } else {
          // std::cout << "Hit case 2 " << cond1 << " " << cond2 << " " << cond3 << std::endl;
          tl.sizes[0] = tl.sizes[loc_tensor_info - 1];
          for (int d = 0; d < depth; d++) tl.addresses[d][0] = tl.addresses[d][loc_tensor_info - 1];
          loc_tensor_info = 1;
          tl.start_tensor_this_launch = t;
        }
      }
    }
  }
}
