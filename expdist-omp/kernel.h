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

template <int tile_size, int stride, typename T>
inline
void fill_shared_mem_tiled_1D(
  T (&sh_mem)[tile_size*stride], 
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
template<typename T, int dim>
inline
void distance_tiled(
  const T *__restrict A, 
  const T *__restrict B,
  int m, int n, 
  const T *__restrict scale_A,
  const T *__restrict scale_B,
  T *__restrict cross_term)
{
  const int nTeamsX = m / (block_size_x * tile_size_x);
  const int nTeams =  nTeamsX * (n / (block_size_y * tile_size_y));
  #pragma omp target teams num_teams(nTeams) thread_limit(block_size_x * block_size_y)
  {
    T sh_A[dim][block_size_x*tile_size_x];
    T sh_B[dim][block_size_y*tile_size_y];
    T sh_scale_A[block_size_x*tile_size_x];
    T sh_scale_B[block_size_y*tile_size_y];
    T sum;
    #pragma omp parallel 
    {
      int tx = omp_get_thread_num() % block_size_x;
      int ty = omp_get_thread_num() / block_size_x;
      int bx = omp_get_team_num() % nTeamsX;
      int by = omp_get_team_num() / nTeamsX;
      int i = tx + bx * block_size_x * tile_size_x;
      int j = ty + by * block_size_y * tile_size_y;

      if (tx == 0 && ty == 0) sum = 0;

      #pragma unroll
      for (int d=0; d<dim; d++) {
        fill_shared_mem_tiled_1D<tile_size_x, block_size_x>(sh_A[d], A+d*m, tx, i);
        fill_shared_mem_tiled_1D<tile_size_y, block_size_y>(sh_B[d], B+d*n, ty, j);
      }
      fill_shared_mem_tiled_1D<tile_size_x, block_size_x>(sh_scale_A, scale_A, tx, i);
      fill_shared_mem_tiled_1D<tile_size_y, block_size_y>(sh_scale_B, scale_B, ty, j);

      T s_cross_term = 0.0;
      #pragma unroll
      for (int ti=0; ti<tile_size_x; ti++) {
        #pragma unroll
        for (int tj=0; tj<tile_size_y; tj++) {

          if ((i+ti*block_size_x < m) && (j+tj*block_size_y < n)) {

            T dist_ij = 0;

            #pragma unroll
            for (int d=0; d<dim; d++) {
              dist_ij += (sh_A[d][tx+ti*block_size_x]-sh_B[d][ty+tj*block_size_y])*
                         (sh_A[d][tx+ti*block_size_x]-sh_B[d][ty+tj*block_size_y]);
            }
            s_cross_term += exp(-dist_ij/(sh_scale_A[tx+ti*block_size_x] + sh_scale_B[ty+tj*block_size_y]));
          }
        }
      }

      #pragma omp atomic update
      sum += s_cross_term;

      #pragma omp barrier

      //write back the per-thread block partial cross term
      if (tx == 0 && ty == 0) {
        cross_term[by * (m / (block_size_x*tile_size_x)) + bx] = sum;
      }
    }
  }
}

template<typename T>
void distance(
  const T *__restrict A,
  const T *__restrict B,
  int m, int n,
  const T *__restrict scale_A,
  const T *__restrict scale_B, 
        T *__restrict cross_term)
{
  //2-dimensional with T precision
  distance_tiled<T, 2>(A, B, m, n, scale_A, scale_B, cross_term);
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
T reduce_cross_term(
  const T *__restrict cross_term,
  int m, int n, int nblocks)
{
  T sum = 0;
  #pragma omp target teams distribute parallel for reduction(+:sum) \
  thread_limit(reduce_block_size) map(tofrom: sum)
  for (int i = 0; i < nblocks; i++)
    sum += cross_term[i];
  return sum;
}
