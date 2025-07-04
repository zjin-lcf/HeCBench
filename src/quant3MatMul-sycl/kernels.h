#include <sycl/sycl.hpp>

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;

template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline T atomicAdd(T &val, const T delta)
{
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     MemoryScope> ref(val);
    return ref.fetch_add(delta);
}

inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

template <typename scalar_t>
void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width,
    const sycl::nd_item<3> &item,
    scalar_t *blockvec)
{
  int row = BLOCKHEIGHT * item.get_group(2);
  int col = BLOCKWIDTH * item.get_group(1) + item.get_local_id(2);
  //if (row >= height || col >= width) return;

  blockvec[item.get_local_id(2)] =
      vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + item.get_local_id(2)];

  scalar_t scale = scales[col];
  scalar_t zero = zeros[col];

  scalar_t res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp1, tmp2, tmp;

  item.barrier(sycl::access::fence_space::local_space);

  while (k < BLOCKWIDTH) { //while (k < min(height, BLOCKWIDTH)) {
    tmp1 = as_unsigned(mat[i]);
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    k += 10;
  }

  atomicAdd(mul[col], res);
}

void VecQuant3MatMulKernelFaster(const sycl::half2 *__restrict__ vec,
                                 const int *__restrict__ mat,
                                 float *__restrict__ mul,
                                 const float *__restrict__ scales,
                                 const float *__restrict__ zeros, int height,
                                 int width, const sycl::nd_item<3> &item,
                                 sycl::half2 *blockvec,
                                 sycl::half2 deq2[64][32])
{
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * item.get_group(2);
  int col = BLOCKWIDTH * item.get_group(1) + item.get_local_id(2);

  if (item.get_local_id(2) < blockwidth2)
    blockvec[item.get_local_id(2)] =
        vec[(row / BLOCKHEIGHT) * blockwidth2 + item.get_local_id(2)];

  int val = item.get_local_id(2) / 32;
  int off = item.get_local_id(2) % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] =
        sycl::half2(sycl::vec<int, 1>(val & 0x7)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0],
                    sycl::vec<int, 1>(val >> 3)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
  }

  sycl::half2 scale =
      sycl::float2(scales[col]).convert<sycl::half, sycl::rounding_mode::rte>();
  sycl::half2 zero =
      sycl::float2(-zeros[col]).convert<sycl::half, sycl::rounding_mode::rte>();

  int i = width * row + col;
  int k = 0;

  float res = 0;
  sycl::half2 res2;

  unsigned int tmp1, tmp2, tmp;

  item.barrier(sycl::access::fence_space::local_space);

  while (k < blockwidth2) {
    res2 = {};
    tmp1 = as_unsigned(mat[i]);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 0) & 0x3f][off], scale, zero),
                     blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 6) & 0x3f][off], scale, zero),
                     blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero),
                     blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero),
                     blockvec[k + 3], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero),
                     blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = sycl::fma(sycl::fma(deq2[tmp][off], scale, zero), blockvec[k + 5],
                     res2);
    tmp2 >>= 4;
    k += 6;
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >> 0) & 0x3f][off], scale, zero),
                     blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >> 6) & 0x3f][off], scale, zero),
                     blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero),
                     blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero),
                     blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = sycl::fma(sycl::fma(deq2[tmp][off], scale, zero), blockvec[k + 4],
                     res2);
    tmp1 >>= 2;
    k += 5;
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 0) & 0x3f][off], scale, zero),
                     blockvec[k + 0], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 6) & 0x3f][off], scale, zero),
                     blockvec[k + 1], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero),
                     blockvec[k + 2], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero),
                     blockvec[k + 3], res2);
    res2 = sycl::fma(sycl::fma(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero),
                     blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += sycl::vec<sycl::half, 1>(res2.x())
               .convert<float, sycl::rounding_mode::automatic>()[0] +
           sycl::vec<sycl::half, 1>(res2.y())
               .convert<float, sycl::rounding_mode::automatic>()[0];
  }

  atomicAdd(mul[col], res);
}

template <typename scalar_t>
void vecquant3matmul(
  sycl::queue &q,
  scalar_t* vec,
  int* mat,
  scalar_t* mul,
  scalar_t* scales,
  scalar_t* zeros,
  int height,
  int width
)
{
  sycl::range<3> gws (1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                      (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT * BLOCKWIDTH);
  sycl::range<3> lws (1, 1, BLOCKWIDTH);

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<scalar_t, 1> blockvec_sm(
        sycl::range<1>(BLOCKWIDTH), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
          VecQuant3MatMulKernel(
            vec, mat, mul, scales, zeros, height, width, item,
            blockvec_sm.template get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

void vecquant3matmul_faster(sycl::queue &q, sycl::half *vec, int *mat, float *mul,
                            float *scales, float *zeros, int height, int width)
{
  sycl::range<3> gws (1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                      (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT * BLOCKWIDTH);
  sycl::range<3> lws (1, 1, BLOCKWIDTH);

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<sycl::half2, 1> blockvec_sm(sycl::range<1>(128), cgh);
    sycl::local_accessor<sycl::half2[64][32], 0> deq2_sm(cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
          VecQuant3MatMulKernelFaster(
              (sycl::half2 *)vec, mat, mul, scales, zeros, height, width,
              item,
              blockvec_sm.get_multi_ptr<sycl::access::decorated::no>().get(),
              deq2_sm);
    });
  });
}
