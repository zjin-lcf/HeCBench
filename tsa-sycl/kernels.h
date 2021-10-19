template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int MARGIN_X, int MARGIN_Y, int BACKWARDS>
inline void trotter_vert_pair(
    T a, T b, 
    int width, int height,
    T &cell_r, T &cell_i, 
    int kx, int ky, int py, 
    //T rl[BLOCK_HEIGHT][BLOCK_WIDTH],
    const accessor<T, 2, sycl_read_write, access::target::local> &rl,
    //T im[BLOCK_HEIGHT][BLOCK_WIDTH]) 
    const accessor<T, 2, sycl_read_write, access::target::local> &im)
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
    //T rl[BLOCK_HEIGHT][BLOCK_WIDTH],
    const accessor<T, 2, sycl_read_write, access::target::local> &rl,
    //T im[BLOCK_HEIGHT][BLOCK_WIDTH]) 
    const accessor<T, 2, sycl_read_write, access::target::local> &im)
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
