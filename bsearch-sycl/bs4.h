template <typename T>
class BS4;

template <typename T>
void bs4 (sycl::queue &q,
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
      sycl::local_accessor<size_t, 1> k (sycl::range<1>(1), cgh);
      cgh.parallel_for<class BS4<T>>(ndr, [=](sycl::nd_item<1> item) {
         size_t gid = item.get_global_id(0);
         size_t lid = item.get_local_id(0);

         if (lid == 0) {
           unsigned nbits = 0;
           while (n >> nbits) nbits++;
           k[0] = 1UL << (nbits - 1);
         }
         item.barrier(sycl::access::fence_space::local_space);

         size_t p = k[0];
         T z = d_z[gid];
         size_t idx = (d_a[p] <= z) ? p : 0;
         while (p >>= 1) {
           size_t r = idx | p;
           size_t w = r < n ? r : n;
           if (z >= d_a[w]) { 
             idx = r;
           }
         }
         d_r[gid] = idx;
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (bs4) " << (time * 1e-9f) / repeat << " (s)\n";
}
