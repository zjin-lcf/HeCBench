#include <iostream>
#include <limits>  // std::numeric_limits
#include <vector>
#include <sycl/sycl.hpp>

typedef sycl::half float16_t;
typedef float float32_t;

using namespace sycl::ext::oneapi::experimental::matrix;

// Element-wise comparison
template <typename T>
void compareEqual(T const *a, T const *b, uint32_t size,
                  double tolerance = 10.0) {
  double max_relative_error = 0.0;

  for (uint32_t i = 0; i < size; i++) {
    auto valA = a[i];
    auto valB = b[i];
    auto relative_error = fabs(valA - valB) / (fabs(valA) + fabs(valB) + 1.0);

    if (relative_error > max_relative_error ||
        relative_error != relative_error) {
      max_relative_error = relative_error;
    }
  }
  auto eps = std::numeric_limits<T>::epsilon();
  if (max_relative_error != max_relative_error ||
      max_relative_error > eps * tolerance) {
    std::cout << "FAILED\n";
  } else {
    std::cout << "PASSED\n";
  }

  std::cout << "Max relative error: " << max_relative_error << std::endl;
}

// Host GEMM validation
void gemm_cpu_h(uint32_t m, uint32_t n, uint32_t k, float16_t const *a,
                float16_t const *b, float32_t const *c, float32_t *d,
                uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
                float32_t alpha, float32_t beta) {
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      float32_t accum = 0.0f;
      for (uint32_t h = 0; h < k; ++h) {
        accum += static_cast<float32_t>(a[i * lda + h]) *
                 static_cast<float32_t>(b[j * ldb + h]);
      }
      d[i * ldd + j] = alpha * accum + beta * c[i * ldc + j];
    }
  }
}

// Matrix data initialization
template <typename DataT>
static inline void fill(DataT *mat, uint32_t m, uint32_t n) {
  srand(m * n);
  auto ld = n;
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      // Ascending order for each neighboring element.
      // Alternate sign for even / odd
      auto value = (i * n + j) % 13;
      mat[i * ld + j] =
          (value % 3) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
    }
  }
}

// : 16 x 16
const int WMMA_M = 16;
const int WMMA_N = 16;

// : multiples of 16.
const int WMMA_K = 16;

// Device warp size
const uint32_t WAVE_SIZE = 32;  // 64

const int NUM_WAVES_X = 4;
const int NUM_WAVES_Y = 4;
const int T_BLOCK_X = NUM_WAVES_X * WAVE_SIZE;
const int T_BLOCK_Y = NUM_WAVES_Y;

void gemm_wmma(uint32_t m, uint32_t n, uint32_t k, float32_t alpha,
               float32_t beta, int32_t repeat, int32_t verify)
{
    // Bounds check
    if((m < (WMMA_M * NUM_WAVES_X) || n < (WMMA_N * NUM_WAVES_Y) || k < WMMA_K)
       || (m % WMMA_M || n % WMMA_N || k % WMMA_K))
    {
        std::cout << "Unsupported size!\n";
        return;
    }

    int lda = k;
    int ldb = k;
    int ldc = n;
    int ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<float16_t> matrixA(m * k);
    std::vector<float16_t> matrixB(k * n);
    std::vector<float32_t> matrixC(m * n);
    // Fill outputs with NaN to catch contamination
    std::vector<float32_t> matrixD(m * n, std::numeric_limits<float32_t>::signaling_NaN());

    fill(matrixA.data(), m, k);
    fill(matrixB.data(), k, n);
    fill(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    const size_t bytesA = matrixA.size() * sizeof(float16_t);
    const size_t bytesB = matrixB.size() * sizeof(float16_t);
    const size_t bytesC = matrixC.size() * sizeof(float32_t);
    const size_t bytesD = matrixD.size() * sizeof(float32_t);

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

    float16_t* d_a = sycl::malloc_device<float16_t>(matrixA.size(), q);
    q.memcpy(d_a, matrixA.data(), bytesA);

    float16_t* d_b = sycl::malloc_device<float16_t>(matrixB.size(), q);
    q.memcpy(d_b, matrixB.data(), bytesB);

    float32_t* d_c = sycl::malloc_device<float32_t>(matrixC.size(), q);
    q.memcpy(d_c, matrixC.data(), bytesC);

    float32_t* d_d = sycl::malloc_device<float32_t>(matrixD.size(), q);
    q.memcpy(d_d, matrixD.data(), bytesD);

    std::cout << "Launching GEMM kernel..." << std::endl;

    sycl::range<2> lws = {T_BLOCK_Y, T_BLOCK_X};
    size_t rx = (m + WMMA_M * NUM_WAVES_X - 1) / (WMMA_M * NUM_WAVES_X) * T_BLOCK_X;
    size_t ry = (n + WMMA_N * NUM_WAVES_Y - 1) / (WMMA_N * NUM_WAVES_Y) * T_BLOCK_Y;
    sycl::range<2> gws = {ry, rx};

    std::chrono::time_point<std::chrono::steady_clock> start, end;

    double elapsedTimeMs;

    for (uint32_t w = 0; w <= repeat; w++) {
      if (w == 1) {
        q.wait();
        start = std::chrono::steady_clock::now();
      }
    
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class mma>(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item)
          [[sycl::reqd_sub_group_size(WAVE_SIZE)]] {
            sycl::sub_group sg = item.get_sub_group();
            uint32_t majorWarp = item.get_global_id(1) / WAVE_SIZE;
            uint32_t minorWarp = item.get_global_id(0);

            auto p_a = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(d_a);
            auto p_b = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(d_b);
            auto p_c = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(d_c);
            auto p_d = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(d_d);

            joint_matrix<sycl::sub_group, float16_t, use::a, WMMA_M, WMMA_K, layout::row_major> sub_a;
            joint_matrix<sycl::sub_group, float16_t, use::b, WMMA_K, WMMA_N, layout::col_major> sub_b;
            joint_matrix<sycl::sub_group, float32_t, use::accumulator, WMMA_M, WMMA_N> sub_c;
            joint_matrix<sycl::sub_group, float32_t, use::accumulator, WMMA_M, WMMA_N> sub_acc;

            joint_matrix_fill(sg, sub_acc, 0.0f);

            uint32_t cRow = majorWarp * WMMA_M;
            uint32_t cCol = minorWarp * WMMA_N;

            if (cRow < m && cCol < n) {
              for (int i = 0; i < k; i += WMMA_K) {
                joint_matrix_load(sg, sub_a, p_a + cRow * lda + i, lda);
                joint_matrix_load(sg, sub_b, p_b + cCol * ldb + i, ldb);
                joint_matrix_mad(sg, sub_acc, sub_a, sub_b, sub_acc);
              }

              joint_matrix_load(sg, sub_c, p_c + cRow * ldc + cCol, ldc, layout::row_major);
              
              joint_matrix_apply(sg, sub_acc, sub_c, [=] (const float32_t &acc, float32_t &c) 
                                                         {c = alpha * acc + beta * c;});

              joint_matrix_store(sg, sub_c, p_d + cRow * ldd + cCol, ldd, layout::row_major);
            }
         });
      });

      if (w == repeat) {
        q.wait(); // throughput
        end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        elapsedTimeMs = time * 1e-6;
      }
    }

    // GEMM flops converge to 2*mnk
    auto gFlops       = 2.0 * static_cast<double>(m) * n * k * 1.0e-9;
    auto tFlopsPerSec = gFlops * repeat / elapsedTimeMs;

    // Echo performance
    std::cout << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, "
              << "alpha, lda, ldb, "
              << "beta, ldc, ldd, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << WMMA_M << ", " << WMMA_N << ", " << WMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
              << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", "
              << tFlopsPerSec << std::endl;

  if (verify) {
    std::cout << "Validating result with reference..." << std::endl;

    q.memcpy(matrixD.data(), d_d, bytesD).wait();

    // Setup and run reference computation
    std::vector<float32_t> matrixD_ref(m * n, std::numeric_limits<float32_t>::signaling_NaN());
    gemm_cpu_h(m,
               n,
               k,
               matrixA.data(),
               matrixB.data(),
               matrixC.data(),
               matrixD_ref.data(),
               lda,
               ldb,
               ldc,
               ldd,
               alpha,
               beta);

    compareEqual<float32_t>(matrixD.data(), matrixD_ref.data(), m * n);
  }

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);
  sycl::free(d_d, q);

  std::cout << "Finished!" << std::endl;
}

void Usage(std::string program_name) {
  // Utility function to display argument usage
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: ";
  std::cout << program_name << "<M> <N> <K> <repeat> <verify>\n\n";
  std::cout
      << "Dense matrix-matrix multiplication: D = alpha * (A * B) + beta * C\n";
  std::cout << "A: M * K, B: K * N, C: M * N, D: M * N\n";
  exit(-1);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    Usage(argv[0]);
  }

  const uint32_t m = atoi(argv[1]);
  const uint32_t n = atoi(argv[2]);
  const uint32_t k = atoi(argv[3]);
  const int32_t repeat = atoi(argv[4]);
  const int32_t verify = atoi(argv[5]);
  gemm_wmma(m, n, k, 0.5f, 2.0f, repeat, verify);

  return 0;
}
