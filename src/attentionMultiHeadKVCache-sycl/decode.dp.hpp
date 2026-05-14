/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <sycl/sycl.hpp>

#include <iostream>

#include "cp_async.dp.hpp"
#include "math.dp.hpp"
#include "pos_enc.dp.hpp"
#include "utils.dp.hpp"
#include "vec_dtypes.dp.hpp"
#include "cascade.dp.hpp"
#include "state.dp.hpp"
#include <cmath>

namespace flashinfer {

DEFINE_HAS_MEMBER(decode_maybe_q_rope_offset)

using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

template <PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem,
          uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
class SingleDecodeWithKVC;

namespace {

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx,
          uint32_t tile_size, typename AttentionVariant, typename Params,
          typename T>
inline void compute_qk(
    const Params &params, AttentionVariant variant, const uint32_t batch_idx,
    const T *smem, const vec_t<float, vec_size> &q_vec,
    const vec_t<float, vec_size> &freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx,
    uint32_t kv_head_idx, float *s, state_t<vec_size> &st, const uint32_t tx,
    const uint32_t ty, const uint32_t tz, const sycl::nd_item<3> &item_ct1) {
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(
          smem + j * bdx * vec_size, freq, kv_idx_base + tz * tile_size + j,
          item_ct1);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset, item_ct1);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx);
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
    st.m = sycl::max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}

/*!
 * \brief Load v tile from shared memory and update local state
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param s A float indicates the pre-softmax attention score
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param st The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
inline void update_local_state(const T *smem, const float *s,
                                        uint32_t compute_stage_idx,
                                        state_t<vec_size> &st, uint32_t tx) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param st The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename AttentionVariant>
inline void sync_state(AttentionVariant variant, state_t<vec_size> &st,
                                float *smem, float *smem_md, const uint32_t tx,
                                const uint32_t ty, const uint32_t tz,
                                const sycl::nd_item<3> &item_ct1) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    if constexpr (variant.use_softmax) {
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      item_ct1.barrier(sycl::access::fence_space::local_space);
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }
    } else {
      item_ct1.barrier(sycl::access::fence_space::local_space);
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          st.o[i] += oz[i];
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single request
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q [num_qo_heads, head_dim] The query matrix
 * \param k [seq_len, num_kv_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_kv_heads, head_dim] The value matrix in kv-cache
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param head_dim A integer indicates the head dimension
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem,
          uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
void SingleDecodeWithKVCacheKernel(const Params params,
                                   const sycl::nd_item<3> &item_ct1,
                                   uint8_t *smem) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const DTypeQ* q = params.q;
  const DTypeKV* k = params.k;
  const DTypeKV* v = params.v;
  //const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t kv_stride_n = params.kv_stride_n;
  const uint32_t kv_stride_h = params.kv_stride_h;
  DTypeO* o = params.o;
  float* lse = params.lse;
  uint32_t kv_chunk_size = params.kv_chunk_size;

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = item_ct1.get_group(1);
  uint32_t qo_head_idx = kv_head_idx * bdy + item_ct1.get_local_id(1);
  uint32_t kv_chunk_idx = item_ct1.get_group(2);
  uint32_t num_qo_heads = params.num_qo_heads;

  AttentionVariant variant(params, /*batch_idx=*/0, smem);
  const uint32_t seq_len = variant.kv_len;
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                          sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                       sizeof(DTypeKV));

  uint32_t tx = item_ct1.get_local_id(2), ty = item_ct1.get_local_id(1),
           tz = item_ct1.get_local_id(0);
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                sycl::pow(rope_rcp_theta,
                          float(2 * ((tx * vec_size + i) % (head_dim / 2))) /
                              float(head_dim));
    }

    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(q + qo_head_idx * q_stride_h,
                                                freq, seq_len - 1, item_ct1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + qo_head_idx * q_stride_h + tx * vec_size);
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = sycl::min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    item_ct1.barrier(sycl::access::fence_space::local_space);
    compute_qk<pos_encoding_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, /*batch_idx=*/0,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
        q_vec, freq, consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz,
        kv_chunk_size, qo_head_idx, kv_head_idx, s, st_local, tx, ty, tz,
        item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    item_ct1.barrier(sycl::access::fence_space::local_space);
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
        st_local, tx);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // load v
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st_local,
                                      reinterpret_cast<float *>(smem), smem_md,
                                      tx, ty, tz, item_ct1);
  if constexpr (variant.use_softmax) {
    st_local.normalize();
  }

  st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  if (lse != nullptr) {
    lse[kv_chunk_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  }
}


/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single request
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 *   for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_kv_heads,
 *   head_dim] for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param o The output matrix, shape: [num_qo_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param pos_encoding_mode The positional encoding mode
 * \param rope_scale The scaling factor used in RoPE Interpolation
 * \param rope_theta The theta used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
void SingleDecodeWithKVCacheDispatched(Params params,
                                       typename Params::DTypeO *tmp,
                                       sycl::queue &stream) try {
  using DTypeKV = typename Params::DTypeKV;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t seq_len = params.kv_len;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
#ifdef __NVPTX__
  auto compute_capacity = GetCudaComputeCapability(stream);
#endif
  static_assert(bdx <= 32U);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads =
        std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeKV)), bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2U * NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * HEAD_DIM * sizeof(DTypeKV) +
          2U * bdy * bdz * sizeof(float);

      if (seq_len <= 256 || tmp == nullptr) {
        // no need to use partition-kv kernel
        sycl::range<3> gws (bdz, num_kv_heads * bdy, bdx);
        sycl::range<3> lws (bdz, bdy, bdx);
        params.kv_chunk_size = seq_len;
        stream.submit([&](sycl::handler &cgh) {
           sycl::local_accessor <uint8_t, 1> smem (smem_size, cgh);
           cgh.parallel_for(sycl::nd_range<3>(gws, lws),
                                   [=](sycl::nd_item<3> item_ct1) {
             SingleDecodeWithKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                           vec_size, bdx, bdy, bdz, AttentionVariant, Params>(
               params, item_ct1,
               smem.get_multi_ptr<sycl::access::decorated::no>().get());
           });
        });
      } else {
        // use partition-kv kernel
        const auto ctx = stream.get_context();
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
        auto kernel = bundle.template get_kernel<SingleDecodeWithKVC<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx, vec_size, bdx, bdy, bdz, AttentionVariant, Params>>();
        uint32_t max_grid_size = kernel.template ext_oneapi_get_info<
            syclex::info::kernel_queue_specific::max_num_work_groups>(
            stream, num_threads, smem_size);

        //FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            //&num_blocks_per_sm, kernel, num_threads, smem_size));
        printf("max_grid_size: %u\n", max_grid_size);
        uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
        uint32_t kv_chunk_size =
            std::max(ceil_div(seq_len, max_num_kv_chunks), uint32_t(256));
        uint32_t num_chunks = ceil_div(seq_len, kv_chunk_size);
        if (num_chunks == 0 ||  num_kv_heads== 0) {
          std::ostringstream err_msg;
          err_msg << "Invalid kernel configuration: nblks=(" << num_chunks << "," << num_kv_heads << ")";
          FLASHINFER_ERROR(err_msg.str());
        }
        sycl::range<3> gws (bdz, bdy * num_kv_heads, bdx * num_chunks);
        sycl::range<3> lws (bdz, bdy, bdx);
        float* tmp_lse = (float*)(tmp + num_chunks * num_qo_heads * HEAD_DIM);
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp;
        params.lse = tmp_lse;
        params.kv_chunk_size = kv_chunk_size;

        stream.submit([&](sycl::handler &cgh) {
           sycl::local_accessor <uint8_t, 1> smem (smem_size, cgh);
           cgh.parallel_for<SingleDecodeWithKVC<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                           vec_size, bdx, bdy, bdz, AttentionVariant, Params>>
             (sycl::nd_range<3>(gws, lws),
                                   [=](sycl::nd_item<3> item_ct1) {
             SingleDecodeWithKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                           vec_size, bdx, bdy, bdz, AttentionVariant, Params>(
               params, item_ct1,
               smem.get_multi_ptr<sycl::access::decorated::no>().get());
           });
        });

        if constexpr (AttentionVariant::use_softmax) {
          MergeStates(tmp, tmp_lse, o, lse, num_chunks, 1, num_qo_heads, HEAD_DIM, stream);
        } else {
          AttentionSum(tmp, o, num_chunks, 1, num_qo_heads, HEAD_DIM, stream);
        }
      }
    });
  });
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
