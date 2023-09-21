#ifndef BIT_VECTOR_KERNELS_CUH
#define BIT_VECTOR_KERNELS_CUH

#define __syncthread() item.barrier(sycl::access::fence_space::local_space)

template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline void atomicAdd(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed, 
     MemoryScope, sycl::access::address_space::global_space> ref(val);
  ref.fetch_add(delta);
}

template <typename T>
inline T warpReduceSum(T val, sycl::nd_item<1> &item, int width = 32) {
  auto sg = item.get_sub_group(); 
  for (int offset = width / 2; offset > 0; offset /= 2)
    val += sycl::shift_group_left(sg, val, offset);
  return val;
}

template<typename T>
inline T blockReduceSum(T val, sycl::local_ptr<T> reductionArray, sycl::nd_item<1> &item) {
  const int sub_group_size = item.get_sub_group().get_local_range().get(0);
  const int lane = item.get_local_id(0) % sub_group_size; 
  const int wid = item.get_local_id(0) / sub_group_size;
  val = warpReduceSum(val, item);
  if (lane == 0) reductionArray[wid] = val;
  __syncthread();
  if (wid == 0) {
    val = (item.get_local_id(0) < WARPSPERBLOCK) ? reductionArray[lane] : 0;
    val = warpReduceSum(val, item, WARPSPERBLOCK);
  }
  return val;
}

// init kernel ---------------------------------------------------------------
template<typename bit_vector_t, typename index_t>
void initFactor(
  bit_vector_t * Ab,
  const index_t height,
  const uint8_t factorDim,
  const uint32_t seed, 
  const float threshold,
  sycl::nd_item<1> &item)
{
  const index_t tid = item.get_global_id(0);

  if(tid < height) {
    bit_vector_t Ai = 0;

    const int randDepth = -sycl::log2((float)threshold) + 1;
    // if threshold very small simply initilize as 0s (also catch threshold=0)
    if(randDepth < 16) {
      fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + tid);

      Ai = ~bit_vector_t(0) >> (32-factorDim);
      for(int d=0; d<randDepth; ++d)
        Ai &= fast_kiss32(state);
    }
    Ab[tid] = Ai;
  }
}


// distance kernels ---------------------------------------------------------------
template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
void computeDistanceRowsShared(
  const bit_factor_t * __restrict Ab,
  const bit_factor_t * __restrict Bb,
  const bit_matrix_t * __restrict Cb, 
  const index_t height,
  const index_t width,
  const index_t padded_width,
  const uint8_t factorDim,
  const error_t weight,
  error_t *global_error,
  sycl::nd_item<1> item,
  sycl::local_ptr<bit_factor_t> B_block,
  sycl::local_ptr<bit_matrix_t> C_block,
  sycl::local_ptr<error_t> reductionArray)
{
  const int sub_group_size = item.get_sub_group().get_local_range().get(0);
  const index_t warpIdIntern = item.get_local_id(0) / sub_group_size;
  const index_t warpId = item.get_group(0) * WARPSPERBLOCK + warpIdIntern;
  const index_t warpLane = item.get_local_id(0) % sub_group_size;
  const index_t blockSize = WARPSPERBLOCK*32;

  const index_t i = warpId;
  const bit_factor_t A_i = i < height ? Ab[i] : 0;

  const index_t vecRow = i / 32;
  const index_t vecFirst = vecRow * padded_width;
  const index_t vecLane = i % 32;
  const index_t col_in_tile = warpLane;
  const index_t padded_width_blocks = SDIV(width, blockSize) * blockSize;
  error_t error_thread = 0;
  for (index_t j = item.get_local_id(0); j < padded_width_blocks;
       j += blockSize) {
    B_block[item.get_local_id(0)] = (j < width) ? Bb[j] : 0;
    C_block[item.get_local_id(0)] = (j < width) ? Cb[vecFirst + j] : 0;
    __syncthread();

    if(i < height) {
      #pragma unroll
      for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
        const bit_factor_t B_j =
            B_block[w * sub_group_size + warpLane];

        const int C_ij =
            (C_block[w * sub_group_size + col_in_tile] >>
             vecLane) &
            1;

        const int product = (B_j & A_i) ? 1 : 0;

        error_thread += error_measure(product, C_ij, weight);
      }
    }
    __syncthread();
  }

  const error_t error_block = blockReduceSum(error_thread, reductionArray, item);
  // Thread with localID == 0 now has total error of block

  if (item.get_local_id(0) == 0)
    atomicAdd(*global_error, error_block);
}


// update kernerls ---------------------------------------------------------------
// [A] row Change kernel
template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
void vectorMatrixMultCompareRowWarpShared(
        bit_factor_t * __restrict A,
  const bit_factor_t * __restrict B,
  const bit_matrix_t * __restrict C,
  const index_t height,
  const index_t width,
  const index_t padded_width,
  const uint8_t factorDim,
  const index_t startrow,
  error_t *global_error,
  const uint32_t seed, 
  const float temperature,
  const float flipManyChance,
  const uint32_t flipManyDepth,
  const error_t weight,
  sycl::nd_item<1> &item,
  sycl::local_ptr<bit_factor_t> B_block,
  sycl::local_ptr<bit_matrix_t> C_block)
{

  const int sub_group_size = item.get_sub_group().get_local_range().get(0);
  const index_t warpId = item.get_group(0) * WARPSPERBLOCK +
                         item.get_local_id(0) / sub_group_size;
  const index_t warpLane = item.get_local_id(0) % sub_group_size;

  const index_t padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
  const index_t i = (startrow + warpId) % padded_height_blocks;

  fast_kiss_state32_t state;

  const bit_factor_t A_i = i < height ? A[i] : 0;
  bit_factor_t A_i_changed = 0;
  if (i < height) {
    state = get_initial_fast_kiss_state32(seed + warpId);

    A_i_changed = A_i ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
  }

  const index_t vecRow = i / 32;
  const index_t vecFirst = vecRow * padded_width;
  const index_t vecLane = i % 32;
  const index_t col_in_tile = warpLane;
  const index_t padded_width_blocks = SDIV(width, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
  error_t error_thread = 0;
  for (index_t j = item.get_local_id(0); j < padded_width_blocks;
       j += WARPSPERBLOCK * 32) {
    B_block[item.get_local_id(0)] = (j < width) ? B[j] : 0;
    C_block[item.get_local_id(0)] = (j < width) ? C[vecFirst + j] : 0;

    __syncthread();

    if(i < height) {
      #pragma unroll
      for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
        const bit_factor_t B_j = B_block[w * sub_group_size + warpLane];
        const int C_ij = (C_block[w * sub_group_size + col_in_tile] >> vecLane) & 1;

        const int product_new = (B_j & A_i_changed) ? 1 : 0;
        const int product_old = (B_j & A_i        ) ? 1 : 0;

        error_thread += error_measure(product_new, C_ij, weight)
          - error_measure(product_old, C_ij, weight);
      }
    }

    __syncthread();
  }
  if(i < height) {
    const error_t error_warp = warpReduceSum(error_thread, item);
    // Thread with warpLane==0 now has total error of warp

    // Thread 0 checks if new low has been found and applies if necessary
    if (warpLane == 0) {
      // Metropolis Hastings algorithm
      if (metro(state, error_warp, temperature, width)) {
        A[i] = A_i_changed;
        atomicAdd(*global_error, error_warp);
      }
    }
  }
}

// [B] col change kernel
template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
void vectorMatrixMultCompareColWarpShared(
  const bit_factor_t * __restrict A,
  bit_factor_t * __restrict B,
  const bit_matrix_t * __restrict C,
  const index_t height,
  const index_t width,
  const index_t padded_width,
  const uint8_t factorDim,
  const index_t startcol,
  error_t *global_error,
  const uint32_t seed,
  const float temperature,
  const float flipManyChance,
  const uint32_t flipManyDepth,
  const error_t weight,
  sycl::nd_item<1> item,
  sycl::local_ptr<bit_factor_t> A_block,
  sycl::local_ptr<bit_matrix_t> C_block)
{
  const int sub_group_size = item.get_sub_group().get_local_range().get(0);
  const index_t warpIdIntern = item.get_local_id(0) / sub_group_size;
  const index_t warpId = item.get_group(0) * WARPSPERBLOCK + warpIdIntern;
  const index_t warpLane = item.get_local_id(0) % sub_group_size;

  const index_t padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
  const index_t j = (startcol + warpId) % padded_width_blocks;

  fast_kiss_state32_t state;

  const bit_factor_t B_j = j < width ? B[j] : 0;
  bit_factor_t B_j_changed = 0;
  if (j < width) {
    state = get_initial_fast_kiss_state32(seed + warpId);

    B_j_changed = B_j ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
  }

  error_t error_thread = 0;
  const index_t vecLane = warpLane;
  const index_t col_in_tile = j % 32;
  const index_t colFirst = j / 32 * 32;
  const index_t padded_height_blocks = SDIV(height, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
  for (index_t i = item.get_local_id(0); i < padded_height_blocks;
       i += WARPSPERBLOCK * 32) {
    A_block[item.get_local_id(0)] = (i < height) ? A[i] : 0;
    const index_t vecRow = i / 32;
    const index_t vecFirst = vecRow * padded_width + colFirst;
    C_block[item.get_local_id(0)] =
        (vecRow < SDIV(height, 32)) ? C[vecFirst + warpLane] : 0;

    __syncthread();

    if (j < width) {
      #pragma unroll
      for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
        const bit_factor_t A_i = A_block[w * sub_group_size + warpLane];
        const int C_ij = (C_block[w * sub_group_size + col_in_tile] >> vecLane) & 1;

        const int product_new = (A_i & B_j_changed) ? 1 : 0;
        const int product_old = (A_i & B_j        ) ? 1 : 0;

        error_thread += error_measure(product_new, C_ij, weight)
          - error_measure(product_old, C_ij, weight);
      }
    }
    __syncthread();
  }
  if (j < width) {
    const error_t error_warp = warpReduceSum(error_thread, item);
    // Thread with warpLane==0 now has total error of warp

    // Thread 0 checks if new low has been found and applies if necessary
    if (warpLane == 0) {
      // Metropolis Hastings algorithm
      if (metro(state, error_warp, temperature, height)) {
        B[j] = B_j_changed;
        atomicAdd(*global_error, error_warp);
      }
    }
  }
}


#endif
