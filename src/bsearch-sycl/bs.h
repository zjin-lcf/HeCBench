template <typename T>
class BS;

template <typename T>
void bs (sycl::queue &q,
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
      cgh.parallel_for<class BS<T>>(ndr, [=](sycl::nd_item<1> item) {
        size_t i = item.get_global_id(0);
         T z = d_z[i];
         size_t low = 0;
         size_t high = n;
          while (high - low > 1) {
            size_t mid = low + (high - low)/2;
            if (z < d_a[mid])
              high = mid;
            else
              low = mid;
          }
          d_r[i] = low;
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (bs) " << (time * 1e-9f) / repeat << " (s)\n";
}
