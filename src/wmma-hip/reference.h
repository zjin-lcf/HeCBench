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
void gemm_cpu_h(uint32_t m, uint32_t n, uint32_t k, fp16_t const *a,
                fp16_t const *b, fp32_t const *c, fp32_t *d,
                uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
                fp32_t alpha, fp32_t beta) {
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      fp32_t accum = 0.0f;
      for (uint32_t h = 0; h < k; ++h) {
        accum += static_cast<fp32_t>(a[i * lda + h]) *
                 static_cast<fp32_t>(b[j * ldb + h]);
      }
      d[i * ldd + j] = alpha * accum + beta * c[i * ldc + j];
    }
  }
}
