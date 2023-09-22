template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int MARGIN_X, int MARGIN_Y, int BACKWARDS>
inline void trotter_vert_pair(
    T a, T b,
    int width, int height,
    T &cell_r, T &cell_i,
    int kx, int ky, int py,
    T rl[BLOCK_HEIGHT][BLOCK_WIDTH],
    T im[BLOCK_HEIGHT][BLOCK_WIDTH])
{
  T peer_r;
  T peer_i;

  const int ky_peer = ky + 1 - 2 * BACKWARDS;
  if (py >= BACKWARDS && py < height - 1 + BACKWARDS && ky >= BACKWARDS && ky < BLOCK_HEIGHT - 1 + BACKWARDS) {
    peer_r = rl[ky_peer][kx];
    peer_i = im[ky_peer][kx];
    rl[ky_peer][kx] = a * peer_r - b * cell_i;
    im[ky_peer][kx] = a * peer_i + b * cell_r;
    cell_r = a * cell_r - b * peer_i;
    cell_i = a * cell_i + b * peer_r;
  }
}

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int MARGIN_X, int MARGIN_Y, int BACKWARDS>
inline void trotter_horz_pair(
    T a, T b,
    int width, int height,
    T &cell_r, T &cell_i,
    int kx, int ky, int px,
    T rl[BLOCK_HEIGHT][BLOCK_WIDTH],
    T im[BLOCK_HEIGHT][BLOCK_WIDTH])
{
  T peer_r;
  T peer_i;

  const int kx_peer = kx + 1 - 2 * BACKWARDS;
  if (px >= BACKWARDS && px < width - 1 + BACKWARDS && kx >= BACKWARDS && kx < BLOCK_WIDTH - 1 + BACKWARDS) {
    peer_r = rl[ky][kx_peer];
    peer_i = im[ky][kx_peer];
    rl[ky][kx_peer] = a * peer_r - b * cell_i;
    im[ky][kx_peer] = a * peer_i + b * cell_r;
    cell_r = a * cell_r - b * peer_i;
    cell_i = a * cell_i + b * peer_r;
  }
}

template<typename T, int STEPS, int BLOCK_X, int BLOCK_Y, int MARGIN_X, int MARGIN_Y, int STRIDE_Y>
void tsa_kernel(
    sycl::nd_item<2> &item,
    T a, T b, int width, int height,
    const T * __restrict p_real,
    const T * __restrict p_imag,
          T * __restrict p2_real,
          T * __restrict p2_imag)
{
  auto g = item.get_group();

  sycl::multi_ptr<T[BLOCK_Y][BLOCK_X], sycl::access::address_space::local_space> rl_ptr =
    sycl::ext::oneapi::group_local_memory_for_overwrite<T[BLOCK_Y][BLOCK_X]>(g);
  T (*rl)[BLOCK_X] = *rl_ptr;

  sycl::multi_ptr<T[BLOCK_Y][BLOCK_X], sycl::access::address_space::local_space> im_ptr =
    sycl::ext::oneapi::group_local_memory_for_overwrite<T[BLOCK_Y][BLOCK_X]>(g);
  T (*im)[BLOCK_X] = *im_ptr;

  int blockIdx_x = item.get_group(1);
  int blockIdx_y = item.get_group(0);
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);

  int px = blockIdx_x * (BLOCK_X - 2 * STEPS * MARGIN_X) + threadIdx_x - STEPS * MARGIN_X;
  int py = blockIdx_y * (BLOCK_Y - 2 * STEPS * MARGIN_Y) + threadIdx_y - STEPS * MARGIN_Y;

  // Read block from global into shared memory
  if (px >= 0 && px < width) {
    #pragma unroll
    for (int i = 0, pidx = py * width + px; i < BLOCK_Y / STRIDE_Y; ++i, pidx += STRIDE_Y * width) {
      if (py + i * STRIDE_Y >= 0 && py + i * STRIDE_Y < height) {
        rl[threadIdx_y + i * STRIDE_Y][threadIdx_x] = p_real[pidx];
        im[threadIdx_y + i * STRIDE_Y][threadIdx_x] = p_imag[pidx];
      }
    }
  }

  group_barrier(g);

  // Place threads along the black cells of a checkerboard pattern
  int sx = threadIdx_x;
  int sy;
  if ((STEPS * MARGIN_X) % 2 == (STEPS * MARGIN_Y) % 2) {
    sy = 2 * threadIdx_y + threadIdx_x % 2;
  } else {
    sy = 2 * threadIdx_y + 1 - threadIdx_x % 2;
  }

  // global y coordinate of the thread on the checkerboard (px remains the same)
  // used for range checks
  int checkerboard_py = blockIdx_y * (BLOCK_Y - 2 * STEPS * MARGIN_Y) + sy - STEPS * MARGIN_Y;

  // keep the fixed black cells on registers, reds are updated in shared memory
  T cell_r[BLOCK_Y / (STRIDE_Y * 2)];
  T cell_i[BLOCK_Y / (STRIDE_Y * 2)];

  // read black cells to registers
  #pragma unroll
  for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
    cell_r[part] = rl[sy + part * 2 * STRIDE_Y][sx];
    cell_i[part] = im[sy + part * 2 * STRIDE_Y][sx];
  }

  // update cells STEPS full steps
  #pragma unroll
  for (int i = 0; i < STEPS; i++) {
    // 12344321
    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 0>(
          a, b, width, height, cell_r[part], cell_i[part],
          sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    group_barrier(g);
    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 0>(
          a, b, width, height, cell_r[part], cell_i[part],
          sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    group_barrier(g);
    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
          a, b, width, height, cell_r[part], cell_i[part],
          sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    group_barrier(g);
    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
          a, b, width, height, cell_r[part], cell_i[part],
          sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    group_barrier(g);

    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
          a, b, width, height, cell_r[part], cell_i[part],
          sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    group_barrier(g);
    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
          a, b, width, height, cell_r[part], cell_i[part],
          sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    group_barrier(g);
    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 0>(
          a, b, width, height, cell_r[part], cell_i[part],
          sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    group_barrier(g);
    #pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
      trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 0>
        (a, b, width, height, cell_r[part], cell_i[part],
         sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    group_barrier(g);
  }

  // write black cells in registers to shared memory
  #pragma unroll
  for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
    rl[sy + part * 2 * STRIDE_Y][sx] = cell_r[part];
    im[sy + part * 2 * STRIDE_Y][sx] = cell_i[part];
  }
  group_barrier(g);

  // discard the halo and copy results from shared to global memory
  sx = threadIdx_x + STEPS * MARGIN_X;
  sy = threadIdx_y + STEPS * MARGIN_Y;
  px += STEPS * MARGIN_X;
  py += STEPS * MARGIN_Y;
  if (sx < BLOCK_X - STEPS * MARGIN_X && px < width) {
    #pragma unroll
    for (int i = 0, pidx = py * width + px; i < BLOCK_Y / STRIDE_Y; ++i, pidx += STRIDE_Y * width) {
      if (sy + i * STRIDE_Y < BLOCK_Y - STEPS * MARGIN_Y && py + i * STRIDE_Y < height) {
        p2_real[pidx] = rl[sy + i * STRIDE_Y][sx];
        p2_imag[pidx] = im[sy + i * STRIDE_Y][sx];
      }
    }
  }
}
