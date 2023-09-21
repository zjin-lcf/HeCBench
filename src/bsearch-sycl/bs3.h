template <typename T>
class BS3;

template <typename T>
void bs3 (sycl::queue &q,
          const size_t aSize,
          const size_t zSize,
          const T *d_a,  // N+1
          const T *d_z,  // T
          size_t *d_r,  // T
          const size_t n,
          const int repeat)
{
  sycl::nd_range<1> ndr{sycl::range<1>(zSize), sycl::range<1>(256)};

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class BS3<T>>(ndr, [=] (sycl::nd_item<1> item) {
        size_t i = item.get_global_id(0);
        unsigned nbits = 0;
        while (n >> nbits) nbits++;
        size_t k = 1ULL << (nbits - 1);
        T z = d_z[i];
        size_t idx = (d_a[k] <= z) ? k : 0;
        while (k >>= 1) {
          size_t r = idx | k;
          size_t w = r < n ? r : n; 
          if (z >= d_a[w]) { 
            idx = r;
          }
        }
        d_r[i] = idx;
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (bs3) " << (time * 1e-9f) / repeat << " (s)\n";
}
