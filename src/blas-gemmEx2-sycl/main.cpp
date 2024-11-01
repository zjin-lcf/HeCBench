#include <sys/time.h>
#include <stdio.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include "utils.h"

//#define FP64_GEMM

using sycl::ext::oneapi::bfloat16;
using sycl::half;

sycl::queue q;
auto engine = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
auto stream = dnnl::sycl_interop::make_stream(engine, q);

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
  *A = sycl::malloc_shared<T> (m * k, q);
  *B = sycl::malloc_shared<T> (k * n, q);
  *C = sycl::malloc_shared<S> (m * n, q);
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
}

template <typename T, typename S>
bool onednn_gemm(
    int m, int n, int k,
    T *A, T *B, S *C,
    S alpha, S beta)
{
  dnnl::memory::data_type AType, BType, CType;
  if (std::is_same<T, double>::value) {
    AType = BType = CType = dnnl::memory::data_type::f64;
  }
  else if (std::is_same<T, float>::value) {
    AType = BType = CType = dnnl::memory::data_type::f32;
  }
  else if (std::is_same<T, half>::value) {
    AType = BType = CType = dnnl::memory::data_type::f16;
  }
  else if (std::is_same<T, bfloat16>::value) {
    AType = BType = CType = dnnl::memory::data_type::bf16;
  }
  else if (std::is_same<T, int8_t>::value) {
    AType = BType = dnnl::memory::data_type::s8;
    CType = dnnl::memory::data_type::s32;
  }
  else {
    printf("Not supported data type.");
    return false;
  }
   
  auto a_md = dnnl::memory::desc({m, k}, AType, dnnl::memory::format_tag::ba);
  auto b_md = dnnl::memory::desc({k, n}, BType, dnnl::memory::format_tag::ba);
  auto c_md = dnnl::memory::desc({m, n}, CType, dnnl::memory::format_tag::ba);

  auto a_mem = dnnl::sycl_interop::make_memory(
    a_md, engine, dnnl::sycl_interop::memory_kind::usm, const_cast<T *>(B));
  auto b_mem = dnnl::sycl_interop::make_memory(
    b_md, engine, dnnl::sycl_interop::memory_kind::usm, const_cast<T *>(A));
  auto c_mem = dnnl::sycl_interop::make_memory(
    c_md, engine, dnnl::sycl_interop::memory_kind::usm, const_cast<S *>(C));

  dnnl::primitive_attr matmul_attr;
  dnnl::matmul::primitive_desc matmul_pd;
  matmul_pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md, matmul_attr);
  auto matmul_prim = dnnl::matmul(matmul_pd);

  matmul_prim.execute(stream, {
    {DNNL_ARG_SRC, a_mem},
    {DNNL_ARG_WEIGHTS, b_mem},
    {DNNL_ARG_DST, c_mem}
  });
  stream.wait();

  return true;
}

template <typename T, typename S>
void test_gemm(const int m, const int n, const int k,
               T *A, T *B, S *C,
               const S alpha, const S beta, int iteration)
{
  double total_time = 0;
  struct timeval start, end;

  for (int i = 0; i < iteration; ++i) {
    gettimeofday(&start, NULL);
    bool success = onednn_gemm(n, // number of rows of matrix A and C
                               m, // number of columns of matrix B and C
                               k, // number of columns of A and rows of B
                               B,
                               A,
                               C,
                               alpha,
                               beta);
    gettimeofday(&end, NULL);
    if (!success) break;
    else if (i > 0) {
      total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
  }
  if (total_time > 0.0) {
    double avg_time = total_time / (iteration - 1);
    printf("%.3f ms\n", avg_time);
    performance(m, n, k, std::is_same<T, int8_t>::value, avg_time * 1e-3);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <M> <N> <K> <iterations>\n", argv[0]);
    printf("C = A X B (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int iteration = atoi(argv[4]);

  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

#ifdef USE_GPU
    q = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    q = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

#ifdef FP64_GEMM
  const double d_alpha = 1.0, d_beta = 0.0;
#endif
  const float f_alpha = 1.f, f_beta = 0.f;
  half h_alpha = sycl::vec<float, 1>{1.f}
                           .convert<half, sycl::rounding_mode::rte>()[0],
       h_beta = sycl::vec<float, 1>{0.f}
                          .convert<half, sycl::rounding_mode::rte>()[0];
  bfloat16 bf_alpha = (bfloat16)1.f;
  bfloat16 bf_beta = (bfloat16)0.f;
  const int32_t i_alpha = 1, i_beta = 0;

#ifdef FP64_GEMM
  double *dA, *dB, *dC;
#endif
  float *fA, *fB, *fC;
  half *hA, *hB, *hC;
  bfloat16 *bfA, *bfB, *bfC;
  int8_t *iA, *iB; int32_t *iC;

#ifdef FP64_GEMM
  allocate_memory(m, n, k, &dA, &dB, &dC);
#endif
  allocate_memory(m, n, k, &fA, &fB, &fC);
  allocate_memory(m, n, k, &hA, &hB, &hC);
  allocate_memory(m, n, k, &bfA, &bfB, &bfC);
  allocate_memory(m, n, k, &iA, &iB, &iC);

  for (int i = 0; i < m * k; ++i) {
#ifdef FP64_GEMM
    dA[i] = double(i % 255 - 127) / 127;
#endif
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = sycl::vec<float, 1>{fA[i]}
                .convert<half, sycl::rounding_mode::rte>()[0];
    bfA[i] = bfloat16(fA[i]);
    iA[i] = float2int8(fA[i], 127);
  }
  for (int i = 0; i < k * n; ++i) {
#ifdef FP64_GEMM
    dB[i] = double(i % 255 - 127) / 127;
#endif
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = sycl::vec<float, 1>{fB[i]}
                .convert<half, sycl::rounding_mode::rte>()[0];
    bfB[i] = bfloat16(fB[i]);
    iB[i] = float2int8(fB[i], 127);
  }

#ifdef FP64_GEMM
  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, dA, dB, dC, d_alpha, d_beta, iteration);
#endif

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, fA, fB, fC, f_alpha, f_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, hA, hB, hC, h_alpha, h_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test bfloat16 >>>>>>>>>>>>>\n");
  test_gemm(m, n, k, bfA, bfB, bfC, bf_alpha, bf_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm(m, n, k, iA, iB, iC, i_alpha, i_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> compare first ten values >>>>>>>>>>>>>>>>>\n");
#ifdef FP64_GEMM
  printf("fp64: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5lf%c", dC[i], " \n"[i==9]);
#endif

  printf("fp32: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fC[i], " \n"[i==9]);

  printf("fp16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hC[i]), " \n"[i==9]);

  printf("bf16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(bfC[i]), " \n"[i==9]);

  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

#ifdef FP64_GEMM
  free_memory(dA, dB, dC);
#endif
  free_memory(fA, fB, fC);
  free_memory(hA, hB, hC);
  free_memory(bfA, bfB, bfC);
  free_memory(iA, iB, iC);
  return 0;
}
