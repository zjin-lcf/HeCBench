#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

/*
* Kernel distributes exact part of the kernel shap dataset
* Each block scatters the data of a row of `observations` into the (number of rows of
* background) in `dataset`, based on the row of `X`.
* So, given:
* background = [[0, 1, 2],
                [3, 4, 5]]
* observation = [100, 101, 102]
* X = [[1, 0, 1],
*      [0, 1, 1]]
*
* dataset (output):
* [[100, 1, 102],
*  [100, 4, 102]
*  [0, 101, 102],
*  [3, 101, 102]]
*
*
*/

template <typename DataT, typename IdxT>
void exact_rows_kernel(float* X, const IdxT nrows_X, const IdxT ncols,
                                  const DataT* background, const IdxT nrows_background,
                                  DataT* dataset, const DataT* observation,
                                  sycl::nd_item<3> item_ct1) {
  // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure data coelescing
  int col = item_ct1.get_local_id(2);
  int row = item_ct1.get_group(2) * ncols;

  while (col < ncols) {
    // Load the X idx for the current column
    int curr_X = (int)X[row + col];

    // Iterate over nrows_background
    for (int row_idx = item_ct1.get_group(2) * nrows_background;
         row_idx < item_ct1.get_group(2) * nrows_background + nrows_background;
         row_idx += 1) {
      if (curr_X == 0) {
        dataset[row_idx * ncols + col] =
          background[(row_idx % nrows_background) * ncols + col];
      } else {
        dataset[row_idx * ncols + col] = observation[col];
      }
    }
    // Increment the column
    col += item_ct1.get_local_range().get(2);
  }
}

/*
* Kernel distributes sampled part of the kernel shap dataset
* The first thread of each block calculates the sampling of `k` entries of `observation`
* to scatter into `dataset`. Afterwards each block scatters the data of a row of `X` into the (number of rows of
* background) in `dataset`.
* So, given:
* background = [[0, 1, 2, 3],
                [5, 6, 7, 8]]
* observation = [100, 101, 102, 103]
* nsamples = [3, 2]
*
* X (output)
*      [[1, 0, 1, 1],
*       [0, 1, 1, 0]]
*
* dataset (output):
* [[100, 1, 102, 103],
*  [100, 6, 102, 103]
*  [0, 101, 102, 3],
*  [5, 101, 102, 8]]
*
*
*/

double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}  

template <typename DataT, typename IdxT>
void sampled_rows_kernel(const IdxT* nsamples, float* X, const IdxT nrows_X,
                                    const IdxT ncols, DataT* background,
                                    const IdxT nrows_background, DataT* dataset,
                                    const DataT* observation, uint64_t seed,
                                    sycl::nd_item<3> item_ct1) {
  // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // see what k this block will generate
  int k_blk = nsamples[item_ct1.get_group(2)];

  // First k threads of block generate samples
  if (item_ct1.get_local_id(2) < k_blk) {
    int rand_idx = (int)(LCG_random_double(&seed) * ncols);

    // Since X is initialized to 0, we quickly check for collisions (if k_blk << ncols the likelyhood of collisions is low)
    while (dpct::atomic_exchange(
               &(X[2 * item_ct1.get_group(2) * ncols + rand_idx]), (float)1) ==
           1) {
      rand_idx = (int)(LCG_random_double(&seed) * ncols);
    }
  }
  item_ct1.barrier();

  // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure data coelescing
  int col_idx = item_ct1.get_local_id(2);
  while (col_idx < ncols) {
    // Load the X idx for the current column
    int curr_X = (int)X[2 * item_ct1.get_group(2) * ncols + col_idx];
    X[(2 * item_ct1.get_group(2) + 1) * ncols + col_idx] = 1 - curr_X;

    for (int bg_row_idx = 2 * item_ct1.get_group(2) * nrows_background;
         bg_row_idx <
         2 * item_ct1.get_group(2) * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx % nrows_background) * ncols + col_idx];
      } else {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      }
    }

    for (int bg_row_idx = (2 * item_ct1.get_group(2) + 1) * nrows_background;
         bg_row_idx <
         (2 * item_ct1.get_group(2) + 1) * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      } else {
        // if(threadIdx.x == 0) printf("tid bg_row_idx: %d %d\n", tid, bg_row_idx);
        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx) % nrows_background * ncols + col_idx];
      }
    }

    col_idx += item_ct1.get_local_range().get(2);
  }
}

template <typename DataT, typename IdxT>
void kernel_dataset(float* X, 
                    const IdxT nrows_X,
                    const IdxT ncols,
                    DataT* background,
                    const IdxT nrows_background,
                    DataT* dataset,
                    DataT* observation,
                    int* nsamples,
                    const int len_samples, 
                    const int maxsample, 
                    const uint64_t seed)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  IdxT nblks;
  IdxT nthreads;

  nthreads = std::min(256, ncols);
  nblks = nrows_X - len_samples;
  printf("nblks = %d len_samples = %d\n", nblks, len_samples );

  if (nblks > 0) {
    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    std::pair<dpct::buffer_t, size_t> X_buf_ct0 =
        dpct::get_buffer_and_offset(X);
    size_t X_offset_ct0 = X_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> background_buf_ct3 =
        dpct::get_buffer_and_offset(background);
    size_t background_offset_ct3 = background_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> dataset_buf_ct5 =
        dpct::get_buffer_and_offset(dataset);
    size_t dataset_offset_ct5 = dataset_buf_ct5.second;
    std::pair<dpct::buffer_t, size_t> observation_buf_ct6 =
        dpct::get_buffer_and_offset(observation);
    size_t observation_offset_ct6 = observation_buf_ct6.second;
    q_ct1.submit([&](sycl::handler &cgh) {
      auto X_acc_ct0 =
          X_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto background_acc_ct3 =
          background_buf_ct3.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto dataset_acc_ct5 =
          dataset_buf_ct5.first.get_access<sycl::access::mode::read_write>(cgh);
      auto observation_acc_ct6 =
          observation_buf_ct6.first.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1,1,nblks * nthreads), sycl::range<3>(1,1,nthreads)),
          [=](sycl::nd_item<3> item_ct1) {
            float *X_ct0 = (float *)(&X_acc_ct0[0] + X_offset_ct0);
            DataT *background_ct3 =
                (DataT *)(&background_acc_ct3[0] + background_offset_ct3);
            DataT *dataset_ct5 =
                (DataT *)(&dataset_acc_ct5[0] + dataset_offset_ct5);
            DataT *observation_ct6 =
                (DataT *)(&observation_acc_ct6[0] + observation_offset_ct6);
            exact_rows_kernel(X_ct0, nrows_X, ncols, background_ct3,
                              nrows_background, dataset_ct5, observation_ct6,
                              item_ct1);
          });
    });
  }

  //CUDA_CHECK(cudaPeekAtLastError());

  // check if random part of the dataset is needed
  if (len_samples > 0) {
    nblks = len_samples / 2;
    // each block does a sample and its compliment
    /*
    DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    {
      std::pair<dpct::buffer_t, size_t> nsamples_buf_ct0 =
          dpct::get_buffer_and_offset(nsamples);
      size_t nsamples_offset_ct0 = nsamples_buf_ct0.second;
      std::pair<dpct::buffer_t, size_t> background_buf_ct4 =
          dpct::get_buffer_and_offset(background);
      size_t background_offset_ct4 = background_buf_ct4.second;
      std::pair<dpct::buffer_t, size_t> observation_buf_ct7 =
          dpct::get_buffer_and_offset(observation);
      size_t observation_offset_ct7 = observation_buf_ct7.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto nsamples_acc_ct0 =
            nsamples_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto background_acc_ct4 =
            background_buf_ct4.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto observation_acc_ct7 =
            observation_buf_ct7.first
                .get_access<sycl::access::mode::read_write>(cgh);

        auto X_nrows_X_len_samples_ncols_ct1 =
            &X[(nrows_X - len_samples) * ncols];
        auto dataset_nrows_X_len_samples_nrows_background_ncols_ct6 =
            &dataset[(nrows_X - len_samples) * nrows_background * ncols];

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1,1,nblks * nthreads), sycl::range<3>(1,1,nthreads)),
            [=](sycl::nd_item<3> item_ct1) {
              int *nsamples_ct0 =
                  (int *)(&nsamples_acc_ct0[0] + nsamples_offset_ct0);
              DataT *background_ct4 =
                  (DataT *)(&background_acc_ct4[0] + background_offset_ct4);
              DataT *observation_ct7 =
                  (DataT *)(&observation_acc_ct7[0] + observation_offset_ct7);
              sampled_rows_kernel(
                  nsamples_ct0, X_nrows_X_len_samples_ncols_ct1, len_samples,
                  ncols, background_ct4, nrows_background,
                  dataset_nrows_X_len_samples_nrows_background_ncols_ct6,
                  observation_ct7, seed, item_ct1);
            });
      });
    }
  }

  //CUDA_CHECK(cudaPeekAtLastError());
}
