/* LUT-GEMM
 * Copyright (c) 2024-present NAVER Cloud Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KERNELS_MV_FP16_HPP
#define KERNELS_MV_FP16_HPP

inline int div_roundup(int x , int y) {
  return (x + y - 1)/ y;
}

// Reference include/dpct/atomic.hpp
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline sycl::half2 atomicAdd(sycl::half2 *addr, sycl::half2 operand) {
  auto atm = sycl::atomic_ref<unsigned, memoryOrder, memoryScope, addressSpace>(
      *reinterpret_cast<unsigned *>(addr));

  union {
    unsigned i;
    sycl::half2 h;
  } old{0}, output{0};

  while (true) {
    old.i = atm.load();
    output.h = old.h + operand;
    if (atm.compare_exchange_strong(old.i, output.i))
      break;
  }

  return output.h;
}

template <int K_TILE_SIZE>
void _nqmv(uint32_t *W, sycl::half *alpha, sycl::half *input,
           sycl::half *output, int M, int K, int NUM_BITS, int M_TILE_SIZE,
           int group_size, const sycl::nd_item<3> &item,
           sycl::half lut[K_TILE_SIZE / 8][256]) {

    const int lut_x_size = item.get_local_range(2) / (K_TILE_SIZE / 8);

    int lut_y = item.get_local_id(2) / lut_x_size;
    int lut_x = item.get_local_id(2) % lut_x_size;

    sycl::half *_inp = &input[item.get_group(1) * K_TILE_SIZE + lut_y * 8];

    sycl::half base =
        +sycl::vec<float, 1>((2 * ((lut_x >> 0) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[0] +
        sycl::vec<float, 1>((2 * ((lut_x >> 1) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[1] +
        sycl::vec<float, 1>((2 * ((lut_x >> 2) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[2] +
        sycl::vec<float, 1>((2 * ((lut_x >> 3) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[3] +
        sycl::vec<float, 1>((2 * ((lut_x >> 4) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[4] +
        sycl::vec<float, 1>((2 * ((lut_x >> 5) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[5] +
        sycl::vec<float, 1>((2 * ((lut_x >> 6) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[6] +
        sycl::vec<float, 1>((2 * ((lut_x >> 7) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[7];

    lut[lut_y][lut_x] = base;

    int s = (lut_x_size==1)  ?0:
            (lut_x_size==2)  ?1:
            (lut_x_size==4)  ?2:
            (lut_x_size==8)  ?3:
            (lut_x_size==16) ?4:
            (lut_x_size==32) ?5:
            (lut_x_size==64) ?6: 
            (lut_x_size==128)?7: 8;  

    for(;s<8;s++){
        sycl::half iValue =
            sycl::vec<float, 1>(2)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] =  lut[lut_y][i +  lut_x - (1 << s)] + iValue;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    int m_start =
        item.get_group(2) * M_TILE_SIZE + item.get_local_id(2) * 2;
    int m_end = (item.get_group(2) + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = item.get_local_range(2) * 2;

    uint32_t *bW = &W[item.get_group(1) * K_TILE_SIZE / 32 * NUM_BITS * M];
    int group_idx = (item.get_group(1) * K_TILE_SIZE) / group_size;
    for(int m = m_start;m < m_end;m += m_step){
        sycl::half reg_o0 = 0;
        sycl::half reg_o1 = 0;
        for(int b=0;b < NUM_BITS;b++){
            sycl::half reg_a0 = alpha[group_idx * NUM_BITS * M + b * M + m + 0];
            sycl::half reg_a1 = alpha[group_idx * NUM_BITS * M + b * M + m + 1];
            sycl::half reg_t_o0 = 0;
            sycl::half reg_t_o1 = 0;
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 
                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + lut[kt*4 + 3][reg_w3]; 
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
        }
        atomicAdd(
            (sycl::half2 *)&output[m], sycl::half2(reg_o0, reg_o1));
    }
}
template <int k_tile_size>
inline void _excute_nqmv(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW,
                         sycl::half *input, int num_thraeds, int m_tile_size) {
    sycl::range<3> gws (1, div_roundup(nqW.kSize, k_tile_size),
                           div_roundup(nqW.mSize, m_tile_size) * num_thraeds);
    sycl::range<3> lws (1, 1, num_thraeds);
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<sycl::half[k_tile_size / 8][256], 0>
                lut_acc(cgh);

            auto nqW_bWeight = nqW.bWeight;
            auto nqW_alpha = (sycl::half *)nqW.alpha;
            auto nqW_mSize = nqW.mSize;
            auto nqW_kSize = nqW.kSize;
            auto nqW_nb = nqW.nb;
            auto nqW_kSize_nqW_num_groups = nqW.kSize / nqW.num_groups;

            cgh.parallel_for(
                sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
                    _nqmv<k_tile_size>(
                        nqW_bWeight, nqW_alpha, input, output,
                        nqW_mSize, nqW_kSize, nqW_nb, m_tile_size,
                        nqW_kSize_nqW_num_groups, item, lut_acc);
                });
        });
    }
}

inline auto _get_excute_nqmv(sycl::queue &q, int num_thraeds, size_t bits, size_t k_tile_idx){
    void (*funcs[])(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW, sycl::half *input,
                    int num_thraeds, int m_tile_size) = {
        _excute_nqmv<32 * 1>, _excute_nqmv<32 * 2>, _excute_nqmv<32 * 3>,
        _excute_nqmv<32 * 4>, _excute_nqmv<32 * 5>, _excute_nqmv<32 * 6>,
        _excute_nqmv<32 * 7>, _excute_nqmv<32 * 8>,
    };
    return funcs[k_tile_idx];
}

inline void nqmv(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW, sycl::half *input,
                 int algo) {
    int k_tile_idx   =     1;
    int m_tile_size  =  2048;
    int num_thraeds  =   256;
    void (*func)(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW, sycl::half *input,
                 int num_thraeds, int m_tile_size) =
        _get_excute_nqmv(q, num_thraeds, nqW.nb, k_tile_idx);
    func(q, output, nqW, input, num_thraeds, m_tile_size);
}

template <int K_TILE_SIZE>
void _nqmv_bias(uint32_t *W, sycl::half *alpha, sycl::half *q_bias,
                sycl::half *input, sycl::half *output, int M, int K,
                int NUM_BITS, int M_TILE_SIZE, int group_size,
                const sycl::nd_item<3> &item,
                sycl::half lut[K_TILE_SIZE / 8][256]) {

    const int lut_x_size = item.get_local_range(2) / (K_TILE_SIZE / 8);

    int lut_y = item.get_local_id(2) / lut_x_size;
    int lut_x = item.get_local_id(2) % lut_x_size;

    sycl::half *_inp = &input[item.get_group(1) * K_TILE_SIZE + lut_y * 8];

    sycl::half base =
        +sycl::vec<float, 1>((2 * ((lut_x >> 0) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[0] +
        sycl::vec<float, 1>((2 * ((lut_x >> 1) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[1] +
        sycl::vec<float, 1>((2 * ((lut_x >> 2) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[2] +
        sycl::vec<float, 1>((2 * ((lut_x >> 3) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[3] +
        sycl::vec<float, 1>((2 * ((lut_x >> 4) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[4] +
        sycl::vec<float, 1>((2 * ((lut_x >> 5) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[5] +
        sycl::vec<float, 1>((2 * ((lut_x >> 6) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[6] +
        sycl::vec<float, 1>((2 * ((lut_x >> 7) & 1) - 1))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[7];

    lut[lut_y][lut_x] = base;

    int s = (lut_x_size==1)  ?0:
            (lut_x_size==2)  ?1:
            (lut_x_size==4)  ?2:
            (lut_x_size==8)  ?3:
            (lut_x_size==16) ?4:
            (lut_x_size==32) ?5:
            (lut_x_size==64) ?6: 
            (lut_x_size==128)?7: 8;  

    for(;s<8;s++){
        sycl::half iValue =
            sycl::vec<float, 1>(2)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0] *
            _inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] =  lut[lut_y][i +  lut_x - (1 << s)] + iValue;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    int m_start =
        item.get_group(2) * M_TILE_SIZE + item.get_local_id(2) * 2;
    int m_end = (item.get_group(2) + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = item.get_local_range(2) * 2;

    uint32_t *bW = &W[item.get_group(1) * K_TILE_SIZE / 32 * NUM_BITS * M];
    int group_idx = (item.get_group(1) * K_TILE_SIZE) / group_size;
    for(int m = m_start;m < m_end;m += m_step){
        sycl::half reg_o0 = 0;
        sycl::half reg_o1 = 0;

        {
            sycl::half reg_a0 = q_bias[group_idx * M + m + 0];
            sycl::half reg_a1 = q_bias[group_idx * M + m + 1];
            sycl::half reg_t_o0 = 0;
            sycl::half reg_t_o1 = 0;
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                reg_t_o0 +=  + lut[kt*4 + 0][255];
                reg_t_o0 +=  + lut[kt*4 + 1][255];
                reg_t_o0 +=  + lut[kt*4 + 2][255];
                reg_t_o0 +=  + lut[kt*4 + 3][255]; 

                reg_t_o1 +=  + lut[kt*4 + 0][255];
                reg_t_o1 +=  + lut[kt*4 + 1][255];
                reg_t_o1 +=  + lut[kt*4 + 2][255];
                reg_t_o1 +=  + lut[kt*4 + 3][255]; 
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
        }   
 
        for(int b=0;b < NUM_BITS;b++){
            sycl::half reg_t_o0 = 0;
            sycl::half reg_t_o1 = 0;

            sycl::half reg_a0 = alpha[group_idx * NUM_BITS * M + b * M + m + 0];
            sycl::half reg_a1 = alpha[group_idx * NUM_BITS * M + b * M + m + 1];
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 
                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + lut[kt*4 + 3][reg_w3]; 
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
        }
        atomicAdd(
            (sycl::half2 *)&output[m], sycl::half2(reg_o0, reg_o1));
    }
}
template <int k_tile_size>
inline void _excute_nqmv_bias(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW,
                              sycl::half *input, int num_thraeds,
                              int m_tile_size) {
    sycl::range<3> gws (1, div_roundup(nqW.kSize, k_tile_size),
                           div_roundup(nqW.mSize, m_tile_size) * num_thraeds);
    sycl::range<3> lws (1, 1, num_thraeds);
    {
        q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<sycl::half[k_tile_size / 8][256], 0>
                lut_acc(cgh);

            auto nqW_bWeight = nqW.bWeight;
            auto nqW_alpha = (sycl::half *)nqW.alpha;
            auto nqW_q_bias = (sycl::half *)nqW.q_bias;
            auto nqW_mSize = nqW.mSize;
            auto nqW_kSize = nqW.kSize;
            auto nqW_nb = nqW.nb;
            auto nqW_kSize_nqW_num_groups = nqW.kSize / nqW.num_groups;

            cgh.parallel_for(
                sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
                                 _nqmv_bias<k_tile_size>(
                                     nqW_bWeight, nqW_alpha,
                                     nqW_q_bias, input, output,
                                     nqW_mSize, nqW_kSize, nqW_nb,
                                     m_tile_size, nqW_kSize_nqW_num_groups,
                                     item, lut_acc);
                             });
        });
    }
}

inline auto _get_excute_nqmv_bias(sycl::queue &q, int num_thraeds, size_t bits, size_t k_tile_idx){
    void (*funcs[])(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW, sycl::half *input,
                    int num_thraeds, int m_tile_size) = {
        _excute_nqmv_bias<32 * 1>, _excute_nqmv_bias<32 * 2>,
        _excute_nqmv_bias<32 * 3>, _excute_nqmv_bias<32 * 4>,
        _excute_nqmv_bias<32 * 5>, _excute_nqmv_bias<32 * 6>,
        _excute_nqmv_bias<32 * 7>, _excute_nqmv_bias<32 * 8>,
    };
    return funcs[k_tile_idx];
}

inline void nqmv_bias(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW, sycl::half *input,
                      int algo) {
    int k_tile_idx   =     0;
    int m_tile_size  =  2048;
    int num_thraeds  =   256;
    void (*func)(sycl::queue &q, sycl::half *output, nQWeight_fp16 &nqW, sycl::half *input,
                 int num_thraeds, int m_tile_size) =
        _get_excute_nqmv_bias(q, num_thraeds, nqW.nb, k_tile_idx);
    func(q, output, nqW, input, num_thraeds, m_tile_size);
}



#endif //KERNELS_MV_FP16_HPP

