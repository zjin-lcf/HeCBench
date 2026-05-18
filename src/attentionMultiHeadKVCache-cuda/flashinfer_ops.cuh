/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <optional>

#include "default_decode_params.cuh"
#include "variants.cuh"
#include "decode.cuh"
#include "exception.h"
#include "layout.cuh"
#include "utils.h"

namespace flashinfer {


template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream);

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t SingleDecodeWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, DTypeO* tmp,
                                    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t seq_len,
                                    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
                                    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                    std::optional<float> maybe_sm_scale = std::nullopt,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr) {
  float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  DISPATCH_HEAD_DIM(
      head_dim, HEAD_DIM, {DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, {
        using Params = SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>;
        using AttentionVariant = DefaultAttention<
            /*use_custom_mask=*/false, /*use_sliding_window=*/false,
            /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
        Params params(q, k, v, o, /*alibi_slopes=*/nullptr, seq_len, num_qo_heads, num_kv_heads,
                      kv_layout, head_dim, /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale,
                      rope_scale, rope_theta);

        SingleDecodeWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant>(
            params, tmp, stream);
      })});
  return cudaSuccess;
}

}  // namespace flashinfer
