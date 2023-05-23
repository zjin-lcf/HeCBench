#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <random>
#include <chrono>
#include <sycl/sycl.hpp>

// Uncomment to use chars as the data type, otherwise use int
// #define CHAR_DATA_TYPE

// Uncomment to use a 4x4 predefined matrix for testing
// #define USE_TEST_MATRIX

#define klog2(n) ((n<8)?2:((n<16)?3:((n<32)?4:((n<64)?5:((n<128)?6:((n<256)?7:((n<512)?8:\
                 ((n<1024)?9:((n<2048)?10:((n<4096)?11:((n<8192)?12:((n<16384)?13:0))))))))))))

#define kmin(x,y) ((x<y)?x:y)
#define kmax(x,y) ((x>y)?x:y)

#define __syncthreads() item.barrier(sycl::access::fence_space::local_space)

inline int atomicExch(int& var, const int val) {
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> (var);
  return atm.exchange(val);
}

inline int atomicAdd(int *var, int val) 
{
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*var);
  return atm.fetch_add(val);
}

#ifndef USE_TEST_MATRIX
#ifdef _n_
// These values are meant to be changed by scripts
const int n = _n_;               // size of the cost/pay matrix
const int rand_range = _range_;  // defines the range of the random matrix.
const int user_n = n;          
const int n_tests = 100;
#else
// User inputs: These values should be changed by the user
const int user_n = 1000;           // This is the size of the cost matrix as supplied by the user
const int n = 1<<(klog2(user_n)+1);// The size of the cost/pay matrix used in the algorithm that is increased to a power of two
const int rand_range = n;          // defines the range of the random matrix.
const int n_tests = 10;            // defines the number of tests performed
#endif

// End of user inputs

const int log2_n = klog2(n);
// Number of threads used in small kernels grid size (typically grid size equal to n) in steps 3ini, 3, 4ini, 4a, 4b, 5a and 5b (64)
const int n_threads = kmin(n,64);    
const int n_threads_reduction = kmin(n, 256); // Number of threads used in the redution kernels in step 1 and 6 (256)
const int n_blocks_reduction = kmin(n, 256);  // Number of blocks used in the redution kernels in step 1 and 6 (256)
// Number of threads used the largest grids sizes (typically grid size equal to n*n) in steps 2 and 6 (256)
const int n_threads_full = kmin(n, 256);
const int seed = 45345; // Initialization for the random number generator

#else
const int n = 4;
const int log2_n = 2;
const int n_threads = 2;
const int n_threads_reduction = 2;
const int n_blocks_reduction = 2;
const int n_threads_full = 2;
#endif

const int n_blocks = n / n_threads;  // Number of blocks used in small kernels grid size (typically grid size equal to n)
const int n_blocks_full = n * n / n_threads_full; // Number of blocks used the largest gris sizes (typically grid size equal to n*n)
const int row_mask = (1 << log2_n) - 1; // Used to extract the row from tha matrix position index (matrices are column wise)
const int nrows = n, ncols = n; // The matrix is square so the number of rows and columns is equal to n
const int max_threads_per_block = 256; // The maximum number of threads per block
const int columns_per_block_step_4 = 512; // Number of columns per block in step 4
// Number of blocks in step 4 and 2
const int n_blocks_step_4 = kmax(n / columns_per_block_step_4, 1);
// The size of a data block. Note that this can be bigger than the matrix size.
const int data_block_size = columns_per_block_step_4 * n;
// log2 of the size of a data block. Note that klog2 cannot handle very large sizes
const int log2_data_block_size = log2_n + klog2(columns_per_block_step_4);

// For the selection of the data type used
#ifndef CHAR_DATA_TYPE
typedef int data;
#define MAX_DATA INT_MAX
#define MIN_DATA INT_MIN
#else
typedef unsigned char data;
#define MAX_DATA 255
#define MIN_DATA 0
#endif

// Host variables start with h_ to distinguish them from the corresponding device variables
// Device variables have no prefix.

#ifndef USE_TEST_MATRIX
data h_cost[ncols][nrows];
#else
data h_cost[n][n] = { { 1, 2, 3, 4 }, { 2, 4, 6, 8 }, { 3, 6, 9, 12 }, { 4, 8, 12, 16 } };
#endif
int h_column_of_star_at_row[nrows];
int h_zeros_vector_size;
int h_n_matches;
bool h_found;
bool h_goto_5;

// -------------------------------------------------------------------------------------
// Device code
// -------------------------------------------------------------------------------------

// initializations
void init(sycl::nd_item<1> &item,
          int *__restrict row_of_star_at_column,
          int *__restrict column_of_star_at_row,
          int *__restrict cover_row,
          int *__restrict cover_column)
{
  int i = item.get_global_id(0);
  //for step 2
  if (i < nrows){
    cover_row[i] = 0;
    column_of_star_at_row[i] = -1;
  }
  if (i < ncols){
    cover_column[i] = 0;
    row_of_star_at_column[i] = -1;
  }
}

// STEP 1.
// a) Subtracting the row by the minimum in each row
const int n_rows_per_block = n / n_blocks_reduction;

void min_in_rows_warp_reduce(volatile data* sdata, int tid) {
  if (n_threads_reduction >= 64 && n_rows_per_block < 64) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 32]));
  if (n_threads_reduction >= 32 && n_rows_per_block < 32) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 16]));
  if (n_threads_reduction >= 16 && n_rows_per_block < 16) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 8]));
  if (n_threads_reduction >= 8 && n_rows_per_block < 8) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 4]));
  if (n_threads_reduction >= 4 && n_rows_per_block < 4) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 2]));
  if (n_threads_reduction >= 2 && n_rows_per_block < 2) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 1]));
}

void calc_min_in_rows(sycl::nd_item<1> &item, data *slack,
                      data *min_in_rows, data *sdata)
{
  unsigned int tid = item.get_local_id(0);
  unsigned int bid = item.get_group(0);
  // One gets the line and column from the blockID and threadID.
  unsigned int l = bid * n_rows_per_block + tid % n_rows_per_block;
  unsigned int c = tid / n_rows_per_block;
  unsigned int i = c * nrows + l;
  const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
  data thread_min = MAX_DATA;

  while (i < n * n) {
    thread_min = sycl::min((int)thread_min, (int)(slack[i]));
    i += gridSize;  // go to the next piece of the matrix...
    // gridSize = 2^k * n, so that each thread always processes the same line or column
  }
  sdata[tid] = thread_min;

  __syncthreads();
  if (n_threads_reduction >= 1024 && n_rows_per_block < 1024) {
    if (tid < 512) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 512]));
    } __syncthreads();
  }
  if (n_threads_reduction >= 512 && n_rows_per_block < 512) {
    if (tid < 256) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 256]));
    } __syncthreads();
  }
  if (n_threads_reduction >= 256 && n_rows_per_block < 256) {
    if (tid < 128) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 128]));
    } __syncthreads();
  }
  if (n_threads_reduction >= 128 && n_rows_per_block < 128) {
    if (tid < 64) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 64]));
    } __syncthreads();
  }
  if (tid < 32) min_in_rows_warp_reduce(sdata, tid);
  if (tid < n_rows_per_block) min_in_rows[bid*n_rows_per_block + tid] = sdata[tid];
}

// a) Subtracting the column by the minimum in each column
const int n_cols_per_block = n / n_blocks_reduction;

void min_in_cols_warp_reduce(volatile data* sdata, int tid) {
  if (n_threads_reduction >= 64 && n_cols_per_block < 64) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 32]));
  if (n_threads_reduction >= 32 && n_cols_per_block < 32) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 16]));
  if (n_threads_reduction >= 16 && n_cols_per_block < 16) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 8]));
  if (n_threads_reduction >= 8 && n_cols_per_block < 8) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 4]));
  if (n_threads_reduction >= 4 && n_cols_per_block < 4) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 2]));
  if (n_threads_reduction >= 2 && n_cols_per_block < 2) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 1]));
}

void calc_min_in_cols(
  sycl::nd_item<1> &item,
  const data *__restrict slack,
        data *__restrict min_in_cols,
        data *__restrict sdata)
{
  unsigned int tid = item.get_local_id(0);
  unsigned int bid = item.get_group(0);
  // One gets the line and column from the blockID and threadID.
  unsigned int c = bid * n_cols_per_block + tid % n_cols_per_block;
  unsigned int l = tid / n_cols_per_block;
  const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
  data thread_min = MAX_DATA;

  while (l < n) {
    unsigned int i = c * nrows + l;
    thread_min = sycl::min((int)thread_min, (int)(slack[i]));
    l += gridSize / n;  // go to the next piece of the matrix...
    // gridSize = 2^k * n, so that each thread always processes the same line or column
  }
  sdata[tid] = thread_min;

  __syncthreads();
  if (n_threads_reduction >= 1024 && n_cols_per_block < 1024) {
    if (tid < 512) { sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 512])); } __syncthreads(); }
  if (n_threads_reduction >= 512 && n_cols_per_block < 512) {
    if (tid < 256) { sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 256])); } __syncthreads(); }
  if (n_threads_reduction >= 256 && n_cols_per_block < 256) {
    if (tid < 128) { sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 128])); } __syncthreads(); }
  if (n_threads_reduction >= 128 && n_cols_per_block < 128) {
    if (tid <  64) { sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 64])); } __syncthreads(); }
  if (tid < 32) min_in_cols_warp_reduce(sdata, tid);
  if (tid < n_cols_per_block) min_in_cols[bid*n_cols_per_block + tid] = sdata[tid];
}

void step_1_row_sub(sycl::nd_item<1> &item, data *__restrict slack, const data *__restrict min_in_rows)
{
  int i = item.get_global_id(0);
  int l = i & row_mask;
  slack[i] = slack[i] - min_in_rows[l];  // subtract the minimum in row from that row
}

void step_1_col_sub(sycl::nd_item<1> &item, data *__restrict slack, const data *__restrict min_in_cols,
                    int *zeros_size_b, int *zeros_size)
{
  int i = item.get_global_id(0);
  int c = i >> log2_n;
  slack[i] = slack[i] - min_in_cols[c]; // subtract the minimum in row from that row

  if (i == 0) *zeros_size = 0;
  if (i < n_blocks_step_4) zeros_size_b[i] = 0;
}

// Compress matrix
void compress_matrix(
  sycl::nd_item<1> &item, 
  const data *__restrict slack,
  int *__restrict zeros,
  int *__restrict zeros_size_b,
  int *__restrict zeros_size)
{
  int i = item.get_global_id(0);

  if (slack[i] == 0) {
    atomicAdd(zeros_size, 1);
    int b = i >> log2_data_block_size;
    int i0 = i & ~(data_block_size - 1);    // == b << log2_data_block_size
    int j = atomicAdd(zeros_size_b + b, 1);
    zeros[i0 + j] = i;
  }
}

// STEP 2
// Find a zero of slack. If there are no starred zeros in its
// column or row star the zero. Repeat for each zero.

// The zeros are split through blocks of data so we run step 2 with several thread blocks and rerun the kernel if repeat was set to true.
void step_2(
  sycl::nd_item<1> &item,
  const int *__restrict zeros,
  const int *__restrict zeros_size_b,
  int *__restrict row_of_star_at_column,
  int *__restrict column_of_star_at_row,
  int *__restrict cover_row,
  int *__restrict cover_column,
  bool *__restrict repeat_kernel,
  bool &s_repeat,
  bool &s_repeat_kernel)
{
  int i = item.get_local_id(0);
  int b = item.get_group(0);

  if (i == 0) s_repeat_kernel = false;

  do {
    __syncthreads();
    if (i == 0) s_repeat = false;
    __syncthreads();

    for (int j = i; j < zeros_size_b[b]; j += item.get_local_range(0))
    {
      int z = zeros[(b << log2_data_block_size) + j];
      int l = z & row_mask;
      int c = z >> log2_n;

      if (cover_row[l] == 0 && cover_column[c] == 0) {
        // thread trys to get the line
        if (!atomicExch(cover_row[l], 1)) {
          // only one thread gets the line
          if (!atomicExch(cover_column[c], 1)) {
            // only one thread gets the column
            row_of_star_at_column[c] = l;
            column_of_star_at_row[l] = c;
          }
          else {
            cover_row[l] = 0;
            s_repeat = true;
            s_repeat_kernel = true;
          }
        }
      }
    }
    __syncthreads();
  } while ((s_repeat));

  if ((s_repeat_kernel)) s_repeat_kernel = true;
}

// STEP 3
// uncover all the rows and columns before going to step 3
void step_3_init(
  sycl::nd_item<1> &item,
  int *__restrict cover_row,
  int *__restrict cover_column,
  int *__restrict n_matches)
{
  int i = item.get_global_id(0);
  cover_row[i] = 0;
  cover_column[i] = 0;
  if (i == 0) *n_matches = 0;
}

// Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum
void step_3(
  sycl::nd_item<1> &item,
  const int *__restrict row_of_star_at_column,
        int *__restrict cover_column,
        int *__restrict n_matches)
{
  int i = item.get_global_id(0);
  if (row_of_star_at_column[i]>=0)
  {
    cover_column[i] = 1;
    atomicAdd(n_matches, 1);
  }
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

void step_4_init(
  sycl::nd_item<1> &item,
  int *__restrict column_of_prime_at_row,
  int *__restrict row_of_green_at_column)
{
  int i = item.get_global_id(0);
  column_of_prime_at_row[i] = -1;
  row_of_green_at_column[i] = -1;
}

void step_4(
  sycl::nd_item<1> &item,
  const int *__restrict zeros,
  const int *__restrict zeros_size_b,
  const int *__restrict column_of_star_at_row,
  int *__restrict cover_row,
  int *__restrict cover_column,
  int *__restrict column_of_prime_at_row,
  bool *__restrict goto_5,
  bool *__restrict repeat_kernel,
  bool &s_found,
  bool &s_goto_5,
  bool &s_repeat_kernel)
{

  volatile int *v_cover_row = cover_row;
  volatile int *v_cover_column = cover_column;

  int i = item.get_local_id(0);
  int b = item.get_group(0);

  if (i == 0) {
    s_repeat_kernel = false;
    s_goto_5 = false;
  }

  do {
    __syncthreads();
    if (i == 0) s_found = false;
    __syncthreads();

    for (int j = i; j < zeros_size_b[b]; j += item.get_local_range(0))
    {
      int z = zeros[(b << log2_data_block_size) + j];
      int l = z & row_mask;
      int c = z >> log2_n;
      int c1 = column_of_star_at_row[l];

      for (int n = 0; n < 10; n++) {

        if (!v_cover_column[c] && !v_cover_row[l]) {
          s_found = true;
          s_repeat_kernel = true;
          column_of_prime_at_row[l] = c;

          if (c1 >= 0) {
            v_cover_row[l] = 1;
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
            v_cover_column[c1] = 0;
          }
          else {
            s_goto_5 = true;
          }
        }
      } // for(int n

    } // for(int j
    __syncthreads();
  } while (s_found && !s_goto_5);

  if (i == 0 && s_repeat_kernel) *repeat_kernel = true;
  if (i == 0 && s_goto_5) *goto_5 = true;
}

/* STEP 5:
Construct a series of alternating primed and starred zeros as follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0(if any).
Let Z2 denote the primed zero in the row of Z1(there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/

// Eliminates joining paths
void step_5a(
  sycl::nd_item<1> &item,
  const int *__restrict row_of_star_at_column,
  const int *__restrict column_of_star_at_row,
  const int *__restrict column_of_prime_at_row,
        int *__restrict row_of_green_at_column)
{
  int i = item.get_global_id(0);

  int r_Z0, c_Z0;

  c_Z0 = column_of_prime_at_row[i];
  if (c_Z0 >= 0 && column_of_star_at_row[i] < 0) {
    row_of_green_at_column[c_Z0] = i;

    while ((r_Z0 = row_of_star_at_column[c_Z0]) >= 0) {
      c_Z0 = column_of_prime_at_row[r_Z0];
      row_of_green_at_column[c_Z0] = r_Z0;
    }
  }
}

// Applies the alternating paths
void step_5b(
  sycl::nd_item<1> &item,
        int *__restrict row_of_star_at_column,
        int *__restrict column_of_star_at_row,
  const int *__restrict row_of_green_at_column)
{
  int j = item.get_global_id(0);

  int r_Z0, c_Z0, c_Z2;

  r_Z0 = row_of_green_at_column[j];

  if (r_Z0 >= 0 && row_of_star_at_column[j] < 0) {

    c_Z2 = column_of_star_at_row[r_Z0];

    column_of_star_at_row[r_Z0] = j;
    row_of_star_at_column[j] = r_Z0;

    while (c_Z2 >= 0) {
      r_Z0 = row_of_green_at_column[c_Z2];  // row of Z2
      c_Z0 = c_Z2;              // col of Z2
      c_Z2 = column_of_star_at_row[r_Z0];    // col of Z4

      // star Z2
      column_of_star_at_row[r_Z0] = c_Z0;
      row_of_star_at_column[c_Z0] = r_Z0;
    }
  }
}

// STEP 6
// Add the minimum uncovered value to every element of each covered
// row, and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered lines.

template <unsigned int blockSize>
void min_warp_reduce(volatile data* sdata, int tid) {
  if (blockSize >= 64) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 32]));
  if (blockSize >= 32) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 16]));
  if (blockSize >= 16) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 8]));
  if (blockSize >= 8) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 4]));
  if (blockSize >= 4) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 2]));
  if (blockSize >= 2) sdata[tid] =
      sycl::min((int)(sdata[tid]), (int)(sdata[tid + 1]));
}

template <unsigned int blockSize>  // blockSize is the size of a block of threads
void min_reduce1(
  volatile const data *__restrict g_idata,
  volatile data *__restrict g_odata,
  unsigned int n,
  sycl::nd_item<1> &item,
  const int *__restrict cover_row,
  const int *__restrict cover_column,
  data *__restrict sdata)
{
  unsigned int tid = item.get_local_id(0);
  unsigned int i = item.get_group(0) * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * item.get_group_range(0);
  sdata[tid] = MAX_DATA;

  while (i < n) {
    int i1 = i;
    int i2 = i + blockSize;
    int l1 = i1 & row_mask;
    int c1 = i1 >> log2_n; 
    data g1;
    if (cover_row[l1] == 1 || cover_column[c1] == 1) g1 = MAX_DATA;
    else g1 = g_idata[i1];
    int l2 = i2 & row_mask;
    int c2 = i2 >> log2_n;
    data g2;
    if (cover_row[l2] == 1 || cover_column[c2] == 1) g2 = MAX_DATA;
    else g2 = g_idata[i2];
    sdata[tid] = sycl::min((int)(sdata[tid]), sycl::min((int)g1, (int)g2));
    i += gridSize;
  }

  __syncthreads();
  if (blockSize >= 1024) {
    if (tid < 512) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 512]));
    } __syncthreads();
  }
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 256]));
    } __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 128]));
    } __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 64]));
    } __syncthreads();
  }
  if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
  if (tid == 0) g_odata[item.get_group(0)] = sdata[0];
}

template <unsigned int blockSize>
void min_reduce2(
  volatile const data *__restrict g_idata,
  volatile data *__restrict g_odata,
  unsigned int n,
  sycl::nd_item<1> &item,
  data *__restrict sdata)
{
  unsigned int tid = item.get_local_id(0);
  unsigned int i = item.get_group(0) * (blockSize * 2) + tid;

  sdata[tid] = sycl::min((int)(g_idata[i]), (int)(g_idata[i + blockSize]));

  __syncthreads();
  if (blockSize >= 1024) {
    if (tid < 512) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 512]));
    } __syncthreads();
  }
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 256]));
    } __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 128]));
    } __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = sycl::min((int)(sdata[tid]), (int)(sdata[tid + 64]));
    } __syncthreads();
  }
  if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
  if (tid == 0) g_odata[item.get_group(0)] = sdata[0];
}

void step_6_add_sub(
  sycl::nd_item<1> &item,
  data *__restrict slack,
  int *__restrict zeros_size_b,
  const int *__restrict cover_row,
  const int *__restrict cover_column,
  const data *__restrict d_min_in_mat,
  int *__restrict zeros_size)
{
  // STEP 6:
  //  /*STEP 6: Add the minimum uncovered value to every element of each covered
  //  row, and subtract it from every element of each uncovered column.
  //  Return to Step 4 without altering any stars, primes, or covered lines. */
  int i = item.get_global_id(0);
  int l = i & row_mask;
  int c = i >> log2_n;
  if (cover_row[l] == 1 && cover_column[c] == 1)
    slack[i] += *d_min_in_mat;
  if (cover_row[l] == 0 && cover_column[c] == 0)
    slack[i] -= *d_min_in_mat;

  if (i == 0) *zeros_size = 0;
  if (i < n_blocks_step_4) zeros_size_b[i] = 0;
}

void min_reduce_kernel1(
  sycl::nd_item<1> &item,
  data *__restrict slack,
  int *__restrict cover_row,
  int *__restrict cover_column,
  data *__restrict d_min_in_mat_vect,
  data *__restrict sdata) 
{
  min_reduce1<n_threads_reduction>(slack, d_min_in_mat_vect, nrows * ncols,
                                   item, cover_row, cover_column, sdata);
}

void min_reduce_kernel2(
  sycl::nd_item<1> &item,
  data *__restrict d_min_in_mat_vect,
  data *__restrict d_min_in_mat,
  data *__restrict sdata) 
{
  min_reduce2<n_threads_reduction / 2>(d_min_in_mat_vect, d_min_in_mat,
                                       n_blocks_reduction, item, sdata);
}

// -------------------------------------------------------------------------------------
// Host code
// -------------------------------------------------------------------------------------

// Used to make sure some constants are properly set
void check(bool val, const char *str){
  if (!val) {
    printf("Check failed: %s!\n", str);
    exit(-1);
  }
}

int main(int argc, char* argv[])
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  if (argc != 2) {
    printf("Usage: %s <output file>\n", argv[0]);
    return 1;
  }

  // Constant checks:
  check(n == (1 << log2_n), "Incorrect log2_n!");
  check(n_threads*n_blocks == n, "n_threads*n_blocks != n\n");
  // step 1
  check(n_blocks_reduction <= n, "Step 1: Should have several lines per block!");
  check(n % n_blocks_reduction == 0, "Step 1: Number of lines per block should be integer!");
  check((n_blocks_reduction*n_threads_reduction) % n == 0,
        "Step 1: The grid size must be a multiple of the line size!");
  check(n_threads_reduction*n_blocks_reduction <= n*n,
        "Step 1: The grid size is bigger than the matrix size!");
  // step 6
  check(n_threads_full*n_blocks_full <= n*n,
        "Step 6: The grid size is bigger than the matrix size!");
  check(columns_per_block_step_4*n == (1 << log2_data_block_size),
        "Columns per block of step 4 is not a power of two!");

  // Open text file
  FILE *file = freopen(argv[1], "w", stdout);
  if (file == NULL)
  {
    perror("Error opening the output file!\n");
    return 1; 
  };

  // Prints the current time
  time_t current_time;
  time(&current_time);
  printf("%s\n", ctime(&current_time));
  fflush(file);

  // total kernel time for all testcases
  long total_time = 0;

  data *slack = sycl::malloc_device<data>(nrows*ncols, q); // The slack matrix
  data *min_in_rows = sycl::malloc_device<data>(nrows, q);  // Minimum in rows
  data *min_in_cols = sycl::malloc_device<data>(ncols, q);  // Minimum in columns
  int *zeros = sycl::malloc_device<int> (nrows*ncols, q); // A vector with the position of the zeros in the slack matrix
  int *zeros_size_b = sycl::malloc_device<int>(n_blocks_step_4, q); // The number of zeros in block i
  int *row_of_star_at_column = sycl::malloc_device<int>(ncols, q); // A vector that given the column j gives the row of the star at that column (or -1, no star)
  int *column_of_star_at_row = sycl::malloc_device<int>(nrows, q); // A vector that given the row i gives the column of the star at that row (or -1, no star)
  int *cover_row = sycl::malloc_device<int>(nrows, q); // A vector that given the row i indicates if it is covered (1- covered, 0- uncovered)
  int *cover_column = sycl::malloc_device<int>(ncols, q); // A vector that given the column j indicates if it is covered (1- covered, 0- uncovered)
  int *column_of_prime_at_row = sycl::malloc_device<int>(nrows, q); // A vector that given the row i gives the column of the prime at that row  (or -1, no prime)
  int *row_of_green_at_column = sycl::malloc_device<int>(ncols, q); // A vector that given the row j gives the column of the green at that row (or -1, no green)
  data *max_in_mat_row = sycl::malloc_device<data>(nrows, q); // Used in step 1 to stores the maximum in rows
  data *min_in_mat_col = sycl::malloc_device<data>(ncols, q); // Used in step 1 to stores the minimums in columns
  data *d_min_in_mat_vect = sycl::malloc_device<data>(n_blocks_reduction, q); // Used in step 6 to stores the intermediate results from the first reduction kernel
  data *d_min_in_mat = sycl::malloc_device<data>(1, q); // Used in step 6 to store the minimum

  int *zeros_size = sycl::malloc_shared<int>(1, q); // The number fo zeros
  int *n_matches = sycl::malloc_shared<int>(1, q); // Used in step 3 to count the number of matches found
  bool *goto_5 = sycl::malloc_shared<bool>(1, q); // After step 4, goto step 5?
  bool *repeat_kernel = sycl::malloc_shared<bool>(1, q); // Needs to repeat the step 2 and step 4 kernel?

#ifndef USE_TEST_MATRIX
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, rand_range-1);

  for (int test = 0; test < n_tests; test++) {
    printf("\n\n\n\ntest %d\n", test);
    fflush(file);

    for (int c = 0; c < ncols; c++)
      for (int r = 0; r < nrows; r++) {
        if (c < user_n && r < user_n)
          h_cost[c][r] = distribution(generator);
        else {
          if (c == r) h_cost[c][r] = 0;
          else h_cost[c][r] = MAX_DATA;
        }
      }
#endif

    // Copy vectors from host memory to device memory
    q.memcpy(slack, h_cost, sizeof(data) * nrows * ncols).wait();

    // Invoke kernels
    auto start = std::chrono::steady_clock::now();

    sycl::range<1> gws1 (n_blocks * n_threads);
    sycl::range<1> lws1 (n_threads);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class init_k>(
        sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
        init(item, 
             row_of_star_at_column,
             column_of_star_at_row,
             cover_row,
             cover_column);
      });
    });

  // Step 1 kernels

    sycl::range<1> gws2 (n_blocks_reduction * n_threads_reduction);
    sycl::range<1> lws2 (n_threads_reduction);
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<data, 1> sm (sycl::range<1>(n_threads_reduction), cgh);
      cgh.parallel_for<class calc_minRow>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        calc_min_in_rows(item, slack, min_in_rows, sm.get_pointer());
      });
    });

  //call_kernel(step_1_row_sub, n_blocks_full, n_threads_full);
    sycl::range<1> gws3 (n_blocks_full * n_threads_full);
    sycl::range<1> lws3 (n_threads_full);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class s1_rowSub>(
        sycl::nd_range<1>(gws3, lws3), [=] (sycl::nd_item<1> item) {
        step_1_row_sub(item, slack, min_in_rows);
      });
    });

  //call_kernel(calc_min_in_cols, n_blocks_reduction, n_threads_reduction);
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<data, 1> sm (sycl::range<1>(n_threads_reduction), cgh);
      cgh.parallel_for<class calc_minCol>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        calc_min_in_cols(item, slack, min_in_cols, sm.get_pointer());
      });
    });

  //call_kernel(step_1_col_sub, n_blocks_full, n_threads_full);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class s1_colSub>(
        sycl::nd_range<1>(gws3, lws3), [=] (sycl::nd_item<1> item) {
        step_1_col_sub(item, slack, min_in_cols, 
            zeros_size_b, zeros_size);
      });
    });

  //call_kernel(compress_matrix, n_blocks_full, n_threads_full);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class compress>(
        sycl::nd_range<1>(gws3, lws3), [=] (sycl::nd_item<1> item) {
        compress_matrix(item, slack, zeros, zeros_size_b, zeros_size);
      });
    });

    // Step 2 kernels
    do {
      repeat_kernel[0] = false;
      //call_kernel(step_2, n_blocks_step_4, (n_blocks_step_4 > 1 || *zeros_size > max_threads_per_block) ? max_threads_per_block : *zeros_size);
      const int block_size = 
        (n_blocks_step_4 > 1 || *zeros_size > max_threads_per_block) ? max_threads_per_block : *zeros_size;
      sycl::range<1> gws4 (n_blocks_step_4 *  block_size);
      sycl::range<1> lws4 (block_size);
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<bool, 0> s_repeat (cgh);
        sycl::local_accessor<bool, 0> s_repeat_kernel (cgh);
        cgh.parallel_for<class s2>(
          sycl::nd_range<1>(gws4, lws4), [=] (sycl::nd_item<1> item) {
          step_2(item, zeros, zeros_size_b, 
              row_of_star_at_column,
              column_of_star_at_row,
              cover_row, cover_column, repeat_kernel, 
              s_repeat,
              s_repeat_kernel);
        });
      }).wait();

      // If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.
    } while (repeat_kernel[0]);

    while (1) {  // repeat steps 3 to 6

      // Step 3 kernels
      //call_kernel(step_3_init, n_blocks, n_threads);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class s3_init>(
          sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
          step_3_init(item, cover_row, cover_column, n_matches);
        });
      });

      //call_kernel(step_3, n_blocks, n_threads);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class s3>(
          sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
          step_3(item, row_of_star_at_column, cover_column, n_matches);
        });
      }).wait();

      if (n_matches[0] >= ncols) break; // It's done

      //step 4_kernels
      //call_kernel(step_4_init, n_blocks, n_threads);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class s4_init>(
          sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
          step_4_init(item, column_of_prime_at_row, row_of_green_at_column);
        });
      });

      while (1) // repeat step 4 and 6
      {
        do {  // step 4 loop
          goto_5[0] = false; repeat_kernel[0] = false;

          //call_kernel(step_4, n_blocks_step_4, (n_blocks_step_4 > 1 || *zeros_size > max_threads_per_block) ? max_threads_per_block : *zeros_size);
          const int block_size = 
            (n_blocks_step_4 > 1 || *zeros_size > max_threads_per_block) ? max_threads_per_block : *zeros_size;
          sycl::range<1> gws4 (n_blocks_step_4 *  block_size);
          sycl::range<1> lws4 (block_size);
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<bool, 0> s_found (cgh);
            sycl::local_accessor<bool, 0> s_goto_5 (cgh);
            sycl::local_accessor<bool, 0> s_repeat_kernel (cgh);
            cgh.parallel_for<class s4>(
              sycl::nd_range<1>(gws4, lws4), [=] (sycl::nd_item<1> item) {
              step_4(item, zeros, zeros_size_b, 
                  column_of_star_at_row,
                  cover_row, 
                  cover_column,
                  column_of_prime_at_row,
                  goto_5,
                  repeat_kernel, 
                  s_found,
                  s_goto_5,
                  s_repeat_kernel);
            });
          }).wait();
          
          // If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.

        } while (repeat_kernel[0] && !goto_5[0]);

        if (goto_5[0]) break;

        //step 6_kernel
        //call_kernel_s(min_reduce_kernel1, n_blocks_reduction, n_threads_reduction, n_threads_reduction*sizeof(int));
        q.submit([&] (sycl::handler &cgh) {
          sycl::local_accessor<int, 1> sm (sycl::range<1>(n_threads_reduction), cgh);
          cgh.parallel_for<class s6_min_reduce>(
            sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
            min_reduce_kernel1(item, slack, 
                cover_row,
                cover_column,
                d_min_in_mat_vect,
                sm.get_pointer());
          });
        });

        //call_kernel_s(min_reduce_kernel2, 1, n_blocks_reduction / 2, (n_blocks_reduction / 2) * sizeof(int));
        sycl::range<1> gws5 (n_blocks_reduction / 2);
        sycl::range<1> lws5 (n_blocks_reduction / 2);
        q.submit([&] (sycl::handler &cgh) {
          sycl::local_accessor<int, 1> sm (sycl::range<1>(n_blocks_reduction/2), cgh);
          cgh.parallel_for<class s6_min_reduce2>(
          sycl::nd_range<1>(gws5, lws5), [=] (sycl::nd_item<1> item) {
            min_reduce_kernel2(item,
                d_min_in_mat_vect,
                d_min_in_mat,
                sm.get_pointer());
          });
        });

        //call_kernel(step_6_add_sub, n_blocks_full, n_threads_full);
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<class s6_addSub>(
          sycl::nd_range<1>(gws3, lws3), [=] (sycl::nd_item<1> item) {
            step_6_add_sub(item, slack,
                      zeros_size_b,
                        cover_row, cover_column, d_min_in_mat,
                        zeros_size);
          });
        });

        //  call_kernel(compress_matrix, n_blocks_full, n_threads_full);
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for<class compress2>(
          sycl::nd_range<1>(gws3, lws3), [=] (sycl::nd_item<1> item) {
            compress_matrix(item, slack, zeros, zeros_size_b, zeros_size);
          });
        });
      } // repeat step 4 and 6

      // call_kernel(step_5a, n_blocks, n_threads);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class s5a>(
          sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
          step_5a(item, row_of_star_at_column,
               column_of_star_at_row, column_of_prime_at_row,
               row_of_green_at_column);
        });
      });

      //call_kernel(step_5b, n_blocks, n_threads);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class s5b>(
          sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
          step_5b(item, row_of_star_at_column,
               column_of_star_at_row, 
               row_of_green_at_column);
        });
      });
    }  // repeat steps 3 to 6

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
    printf("Total kernel execution time of the Hungarian algorithm %f (s)\n", time * 1e-9f);

    fflush(file);

    // Copy assignments from Device to Host and calculate the total Cost
    q.memcpy(h_column_of_star_at_row, column_of_star_at_row, nrows * sizeof(int)).wait();

    int total_cost = 0;
    for (int r = 0; r < nrows; r++) {
      int c = h_column_of_star_at_row[r];
      if (c >= 0) total_cost += h_cost[c][r];
    }

    printf("Total cost is \t %d \n", total_cost);

#ifndef USE_TEST_MATRIX
  }
#endif

  fclose(file);
  fprintf(stderr, "Total kernel time for all test cases %lf (s)\n", total_time * 1e-9);

  sycl::free(slack, q);
  sycl::free(min_in_rows, q);
  sycl::free(min_in_cols, q);
  sycl::free(zeros, q);
  sycl::free(zeros_size_b, q);
  sycl::free(row_of_star_at_column, q);
  sycl::free(column_of_star_at_row, q);
  sycl::free(cover_row, q);
  sycl::free(cover_column, q);
  sycl::free(column_of_prime_at_row, q);
  sycl::free(row_of_green_at_column, q);
  sycl::free(max_in_mat_row, q);
  sycl::free(min_in_mat_col, q);
  sycl::free(d_min_in_mat_vect, q);
  sycl::free(d_min_in_mat, q);
  sycl::free(zeros_size, q);
  sycl::free(n_matches, q);
  sycl::free(goto_5, q);
  sycl::free(repeat_kernel, q);

  return 0;
}
