double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

inline float atomicExch(float& addr, const float val) {
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(addr);
  return ref.exchange(val);
}

template <typename DataT, typename IdxT>
void sampled_rows_kernel(
  sycl::nd_item<1> &item,
  const IdxT*__restrict__ nsamples,
  float*__restrict__ X,
  const IdxT nrows_X,
  const IdxT ncols,
  DataT*__restrict__ background,
  const IdxT nrows_background,
  DataT*__restrict__ dataset,
  const DataT*__restrict__ observation,
  uint64_t seed)
{
  int gid = item.get_group(0);
  int lid = item.get_local_id(0);

  // see what k this block will generate
  int k_blk = nsamples[gid];

  // First k threads of block generate samples
  if (lid < k_blk) {
    int rand_idx = (int)(LCG_random_double(&seed) * ncols);

    // Since X is initialized to 0, we quickly check for collisions (if k_blk << ncols the likelyhood of collisions is low)
    while (atomicExch(X[2 * gid * ncols + rand_idx], 1) == 1) {
      rand_idx = (int)(LCG_random_double(&seed) * ncols);
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure data coelescing
  int col_idx = lid;
  while (col_idx < ncols) {
    // Load the X idx for the current column
    int curr_X = (int)X[2 * gid * ncols + col_idx];
    X[(2 * gid + 1) * ncols + col_idx] = 1 - curr_X;

    for (int bg_row_idx = 2 * gid * nrows_background;
         bg_row_idx < 2 * gid * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx % nrows_background) * ncols + col_idx];
      } else {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      }
    }

    for (int bg_row_idx = (2 * gid + 1) * nrows_background;
         bg_row_idx <
         (2 * gid + 1) * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      } else {
        // if(lid == 0) printf("tid bg_row_idx: %d %d\n", tid, bg_row_idx);
        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx) % nrows_background * ncols + col_idx];
      }
    }

    col_idx += item.get_local_range(0);
  }
}

template <typename T, typename IdxT>
void kernel_dataset(sycl::queue &q,
                    float *d_X,
                    const IdxT nrows_X,
                    const IdxT ncols,
                    T *d_background,
                    const IdxT nrows_background,
                    T *d_dataset,
                    T *d_observation,
                    int *d_nsamples,
                    const int len_samples,
                    const int maxsample,
                    uint64_t seed,
                    double &time)
{
  IdxT nblks;
  IdxT nthreads;

  nthreads = std::min(256, ncols);
  nblks = nrows_X - len_samples;

  sycl::range<1> exact_gws (nblks * nthreads);
  sycl::range<1> lws (nthreads);

  // printf("nblks = %d len_samples = %d\n", nblks, len_samples );

  auto start = std::chrono::steady_clock::now();

  if (nblks > 0) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class exact_rows>(
        sycl::nd_range<1>(exact_gws, lws), [=] (sycl::nd_item<1> item) {
        // Each block processes one row of X.
        // Columns are iterated over by blockDim.x at a time to ensure data coelescing
        int gid = item.get_group(0);
        int col = item.get_local_id(0);
        int row = gid * ncols;

        while (col < ncols) {
          // Load the X idx for the current column
          int curr_X = (int)d_X[row + col];

          // Iterate over nrows_background
          for (int row_idx = gid * nrows_background;
               row_idx < gid * nrows_background + nrows_background;
               row_idx += 1) {
            if (curr_X == 0) {
              d_dataset[row_idx * ncols + col] =
                d_background[(row_idx % nrows_background) * ncols + col];
            } else {
              d_dataset[row_idx * ncols + col] = d_observation[col];
            }
          }
          // Increment the column
          col += item.get_local_range(0);
        }
      });
    });
  }

  // check if random part of the dataset is needed
  if (len_samples > 0) {
    nblks = len_samples / 2;
    sycl::range<1> sampled_gws (nblks * nthreads);

    // each block does a sample and its compliment
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class sampled_rows>(
        sycl::nd_range<1>(sampled_gws, lws), [=] (sycl::nd_item<1> item) {
        sampled_rows_kernel(item,
          d_nsamples, d_X + (nrows_X - len_samples) * ncols,
          len_samples, ncols,
          d_background, nrows_background,
          d_dataset + (nrows_X - len_samples) * nrows_background * ncols,
          d_observation,
          seed);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
