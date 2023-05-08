#ifndef block_size_x
#define block_size_x 32
#endif
#ifndef block_size_y
#define block_size_y 8
#endif

#ifndef tile_size_x
#define tile_size_x 4
#endif
#ifndef tile_size_y
#define tile_size_y 4
#endif

#ifndef reduce_block_size
#define reduce_block_size 256
#endif

#define syncthreads() item.barrier(sycl::access::fence_space::local_space)

template <typename T>
class computeCost;

template <typename T>
class reduceBlock;

template <int tile_size, int stride, typename T>
inline void fill_shared_mem_tiled_1D(
        T *__restrict sh_mem,
  const T *__restrict d_mem,
  int sh_offset, int d_offset)
{
  #pragma unroll
  for (int ti=0; ti<tile_size; ti++) {
    sh_mem[sh_offset+ti*stride] = d_mem[d_offset+ti*stride];
  }
}

/*
 * This function performs the main body of work for computing the Bhattacharya
 * cost function for two given point sets.
 * The parallelization is such that a 2D array of 2D thread blocks are created
 * to match the m*n iteration space. The amount of work per thread is controlled
 * through tiling factors tile_size_x and tile_size_y.
 * The cross term is reduced to a single value per thread block, which then needs
 * to be reduced to a single value in a second kernel. 
 */
#define sh_A_base (d*block_size_x*tile_size_x)
#define sh_B_base (d*block_size_y*tile_size_y)

template<typename T>
void distance_tiled(
  sycl::nd_item<2> &item,
  const T *__restrict A, 
  const T *__restrict B,
  int m, int n, 
  const T *__restrict scale_A,
  const T *__restrict scale_B,
  T *__restrict cross_term,
  T *__restrict sh_A, 
  T *__restrict sh_B,
  T *__restrict sh_scale_A,
  T *__restrict sh_scale_B,
  T *__restrict sum)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int bx = item.get_group(1);
  int by = item.get_group(0);
  int dimx = item.get_group_range(1);
  int i = tx + bx * block_size_x * tile_size_x;
  int j = ty + by * block_size_y * tile_size_y;

  if (tx == 0 && ty == 0) sum[0] = 0;

  #pragma unroll
  for (int d=0; d<2; d++) {
    fill_shared_mem_tiled_1D<tile_size_x, block_size_x>(
       sh_A+sh_A_base, A+d*m, tx, i);

    fill_shared_mem_tiled_1D<tile_size_y, block_size_y>(
       sh_B+sh_B_base, B+d*n, ty, j);
  }
  fill_shared_mem_tiled_1D<tile_size_x, block_size_x>(sh_scale_A, scale_A, tx, i);
  fill_shared_mem_tiled_1D<tile_size_y, block_size_y>(sh_scale_B, scale_B, ty, j);

  T s_cross_term = 0.0;
  #pragma unroll
  for (int ti=0; ti<tile_size_x; ti++) {
    #pragma unroll
    for (int tj=0; tj<tile_size_y; tj++) {

      if ((i+ti*block_size_x < m) && (j+tj*block_size_y < n)) {

        T dist_ij = 0.0;

        #pragma unroll
        for (int d=0; d<2; d++) {
          dist_ij += (sh_A[sh_A_base+(tx+ti*block_size_x)]-sh_B[sh_B_base+(ty+tj*block_size_y)])*
                     (sh_A[sh_A_base+(tx+ti*block_size_x)]-sh_B[sh_B_base+(ty+tj*block_size_y)]);
        }
        s_cross_term += sycl::exp(-dist_ij/(sh_scale_A[tx+ti*block_size_x] + sh_scale_B[ty+tj*block_size_y]));
      }
    }
  }

  auto sum_ref = sycl::atomic_ref<T, 
                 sycl::memory_order::relaxed,
                 sycl::memory_scope::work_group,
                 sycl::access::address_space::local_space> (sum[0]);
  sum_ref.fetch_add(s_cross_term);
  syncthreads();

  //write back the per-thread block partial cross term
  if (tx == 0 && ty == 0) {
    cross_term[by*dimx+bx] = sum[0];
  }
}

/*
 * Reduce the per thread block cross terms computed in the GaussTransform kernel to single value
 *
 * This kernel is designed to run as single-thread block, because the number of terms to reduce is
 * of size n or m, which is expected to be around 2000 or so. The number of items to reduce
 * is passed as the last argument 'nblocks', which corresponds to the number of thread blocks used
 * by the first kernel.
 */
template<typename T>
void reduce_cross_term(
  sycl::nd_item<1> &item,
        T *__restrict output,
  const T *__restrict cross_term,
        T *__restrict sum,
  int m, int n, int nblocks)
{
  int tx = item.get_local_id(0);

  if (tx == 0) sum[0] = 0;
  syncthreads();

  T s_cross_term = 0;
  for (int i=tx; i<nblocks; i+=reduce_block_size)
    s_cross_term += cross_term[i];

  auto sum_ref = sycl::atomic_ref<T, 
                 sycl::memory_order::relaxed,
                 sycl::memory_scope::work_group,
                 sycl::access::address_space::local_space> (sum[0]);
  sum_ref.fetch_add(s_cross_term);

  syncthreads();

  if (tx == 0) output[0] = sum[0];
}
