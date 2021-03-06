
double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}  

template <typename T, typename IdxT>
void kernel_dataset(queue &q,
                    buffer<float,1> &d_X, 
                    const IdxT nrows_X,
                    const IdxT ncols,
                    buffer<T, 1> &d_background,
                    const IdxT nrows_background,
                    buffer<T,1> &d_dataset,
                    buffer<T,1> &d_observation,
                    buffer<int,1> &d_nsamples,
                    const int len_samples, 
                    const int maxsample, 
                     uint64_t seed) 
{

  IdxT nblks;
  IdxT nthreads;

  nthreads = std::min(256, ncols);
  nblks = nrows_X - len_samples;

  range<1> exact_gws (nblks * nthreads);
  range<1> lws (nthreads);

  printf("nblks = %d len_samples = %d\n", nblks, len_samples );

  if (nblks > 0) {
    q.submit([&] (handler &cgh) {
    auto X = d_X.template get_access<sycl_read>(cgh);
    auto background = d_background.template get_access<sycl_read>(cgh);
    auto observation = d_observation.template get_access<sycl_read>(cgh);
    auto dataset = d_dataset.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class exact_rows>(nd_range<1>(exact_gws, lws), [=] (nd_item<1> item) {
     // Each block processes one row of X. 
     // Columns are iterated over by blockDim.x at a time to ensure data coelescing
     int gid = item.get_group(0);
     int col = item.get_local_id(0);
     int row = gid * ncols;

     while (col < ncols) {
       // Load the X idx for the current column
       int curr_X = (int)X[row + col];

       // Iterate over nrows_background
       for (int row_idx = gid * nrows_background;
            row_idx < gid * nrows_background + nrows_background;
            row_idx += 1) {
         if (curr_X == 0) {
           dataset[row_idx * ncols + col] =
             background[(row_idx % nrows_background) * ncols + col];
         } else {
           dataset[row_idx * ncols + col] = observation[col];
         }
       }
       // Increment the column
       col += item.get_local_range(0);
     }
    }); 
    });
  }

  //CUDA_CHECK(cudaPeekAtLastError());

  // check if random part of the dataset is needed
  if (len_samples > 0) {
    nblks = len_samples / 2;
    range<1> sampled_gws (nblks * nthreads);

    // each block does a sample and its compliment
    //sampled_rows_kernel<<<nblks, nthreads>>>(
    //  nsamples, &X[(nrows_X - len_samples) * ncols], len_samples, ncols,
    // background, nrows_background,
    //  &dataset[(nrows_X - len_samples) * nrows_background * ncols], observation,
    //  seed);

    q.submit([&] (handler &cgh) {
    auto X = d_X.template get_access<sycl_atomic>(cgh, len_samples*ncols, (nrows_X - len_samples) * ncols);
    auto background = d_background.template get_access<sycl_read>(cgh);
    auto observation = d_observation.template get_access<sycl_read>(cgh);
    auto dataset = d_dataset.template get_access<sycl_write>(cgh, 
                   len_samples * nrows_background * ncols,
                   (nrows_X - len_samples) * nrows_background * ncols);
    auto nsamples = d_nsamples.template get_access<sycl_read>(cgh);
    cgh.parallel_for<class sampled_rows>(nd_range<1>(sampled_gws, lws), [=] (nd_item<1> item) {
      int gid = item.get_group(0);
      int lid = item.get_local_id(0);
      // int tid = lid + gid * blockDim.x;
      
      // see what k this block will generate
      int k_blk = nsamples[gid];

      // First k threads of block generate samples
      if (lid < k_blk) {
        int rand_idx = (int)(LCG_random_double(const_cast<uint64_t*>(&seed)) * ncols);

        // Since X is initialized to 0, we quickly check for collisions (if k_blk << ncols the likelyhood of collisions is low)
        while (atomic_exchange(X[2 * gid * ncols + rand_idx], (T)1) == (T)1) {
          rand_idx = (int)(LCG_random_double(const_cast<uint64_t*>(&seed)) * ncols);
        }
      }

      item.barrier(access::fence_space::local_space);

      // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure data coelescing
      int col_idx = lid;
      while (col_idx < ncols) {
        // Load the X idx for the current column
        int curr_X = (int)atomic_load(X[2 * gid * ncols + col_idx]);
        atomic_store(X[(2 * gid + 1) * ncols + col_idx], (T)(1 - curr_X));

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
    });
    });
  }
}
