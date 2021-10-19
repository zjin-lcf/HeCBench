#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "common.h"
#include "kernels.h"
#include "reference.h"

// Forward declarations
template<typename T, int STEPS, int BLOCK_X, int BLOCK_Y, int MARGIN_X, int MARGIN_Y, int STRIDE_Y>
class k;

template <typename T>
static void init_p(T *p_real, T *p_imag, int width, int height) {
  double s = 64.0;
  for (int j = 1; j <= height; j++) {
    for (int i = 1; i <= width; i++) {
      // p(i,j)=exp(-((i-180.)**2+(j-300.)**2)/(2*s**2))*exp(im*0.4*(i+j-480.))
      std::complex<T> tmp = std::complex<T>(
        exp(-(pow(i - 180.0, 2.0) + pow(j - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0) *
        exp(std::complex<T>(0.0, 0.4 * (i + j - 480.0)));

      p_real[(j-1) * width + i-1] = real(tmp);
      p_imag[(j-1) * width + i-1] = imag(tmp);
    }
  }
}

template <typename T>
void tsa(queue &q, int width, int height, int repeat) {

  T * p_real = new T[width * height];
  T * p_imag = new T[width * height];
  T * h_real = new T[width * height];
  T * h_imag = new T[width * height];

  // initialize p_real and p_imag matrices
  init_p(p_real, p_imag, width, height);

  // precomputed values
  T a = cos(0.02);
  T b = sin(0.02);

  // compute reference results
  memcpy(h_imag, p_imag, sizeof(T)*width*height);
  memcpy(h_real, p_real, sizeof(T)*width*height);
  reference(h_real, h_imag, a, b, width, height, repeat);

  // thread block / shared memory block width
  static const int BLOCK_X = 16;
  // shared memory block height
  static const int BLOCK_Y = sizeof(T) == 8 ? 32 : 96;
  // thread block height
  static const int STRIDE_Y = 16;

  // halo sizes on each side
  static const int MARGIN_X = 3;
  static const int MARGIN_Y = 4;

  // time step
  static const int STEPS = 1;

  range<2> gws ((height + (BLOCK_Y - 2 * STEPS * MARGIN_Y) - 1) / (BLOCK_Y - 2 * STEPS * MARGIN_Y) * STRIDE_Y,
                (width + (BLOCK_X - 2 * STEPS * MARGIN_X) - 1) / (BLOCK_X - 2 * STEPS * MARGIN_X)  * BLOCK_X);
             
  range<2> lws (STRIDE_Y, BLOCK_X);
  int sense = 0;

  // ping-pong arrays
  std::vector<buffer<T, 1>> d_real;
  std::vector<buffer<T, 1>> d_imag;
  d_real.emplace_back(buffer<T, 1>(p_real, width * height));
  d_real[0].set_final_data(nullptr);
  d_real.emplace_back(buffer<T, 1>(width * height));

  d_imag.emplace_back(buffer<T, 1>(p_imag, width * height));
  d_imag[0].set_final_data(nullptr);
  d_imag.emplace_back(buffer<T, 1>(width * height));

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto p_real = d_real[sense].template get_access<sycl_read>(cgh);
      auto p_imag = d_imag[sense].template get_access<sycl_read>(cgh);
      auto p2_real = d_real[1-sense].template get_access<sycl_discard_write>(cgh);
      auto p2_imag = d_imag[1-sense].template get_access<sycl_discard_write>(cgh);
      accessor<T, 2, sycl_read_write, access::target::local> rl({BLOCK_Y, BLOCK_X}, cgh);
      accessor<T, 2, sycl_read_write, access::target::local> im({BLOCK_Y, BLOCK_X}, cgh);
      cgh.parallel_for<class k<T, STEPS, BLOCK_X, BLOCK_Y, MARGIN_X, MARGIN_Y, STRIDE_Y>>(
        nd_range<2>(gws, lws), [=] (nd_item<2> item) {
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

        item.barrier(access::fence_space::local_space);

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

          #pragma unroll
        // read black cells to registers
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
          item.barrier(access::fence_space::local_space);
          #pragma unroll
          for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
            trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 0>(
                a, b, width, height, cell_r[part], cell_i[part],
                sx, sy + part * 2 * STRIDE_Y, px, rl, im);
          }
          item.barrier(access::fence_space::local_space);
          #pragma unroll
          for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
            trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
                a, b, width, height, cell_r[part], cell_i[part], 
                sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
          }
          item.barrier(access::fence_space::local_space);
          #pragma unroll
          for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
            trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
                a, b, width, height, cell_r[part], cell_i[part],
                sx, sy + part * 2 * STRIDE_Y, px, rl, im);
          }
          item.barrier(access::fence_space::local_space);

          #pragma unroll
          for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
            trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
                a, b, width, height, cell_r[part], cell_i[part],
                sx, sy + part * 2 * STRIDE_Y, px, rl, im);
          }
          item.barrier(access::fence_space::local_space);
          #pragma unroll
          for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
            trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 1>(
                a, b, width, height, cell_r[part], cell_i[part],
                sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
          }
          item.barrier(access::fence_space::local_space);
          #pragma unroll
          for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
            trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 0>(
                a, b, width, height, cell_r[part], cell_i[part],
                sx, sy + part * 2 * STRIDE_Y, px, rl, im);
          }
          item.barrier(access::fence_space::local_space);
          #pragma unroll
          for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
            trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS * MARGIN_X, STEPS * MARGIN_Y, 0>
              (a, b, width, height, cell_r[part], cell_i[part], 
               sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
          }
          item.barrier(access::fence_space::local_space);
        }

        // write black cells in registers to shared memory
          #pragma unroll
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          rl[sy + part * 2 * STRIDE_Y][sx] = cell_r[part];
          im[sy + part * 2 * STRIDE_Y][sx] = cell_i[part];
        }
        item.barrier(access::fence_space::local_space);

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
      });
    });
    sense = 1 - sense; // swap
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_real[sense].template get_access<sycl_read>(cgh);
    cgh.copy(acc, p_real);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_imag[sense].template get_access<sycl_read>(cgh);
    cgh.copy(acc, p_imag);
  });

  q.wait();

  // verify
  bool ok = true;
  for (int i = 0; i < width * height; i++) {
    if (fabs(p_real[i] - h_real[i]) > 1e-3) {
      ok = false;
      break;
    }
    if (fabs(p_imag[i] - h_imag[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] p_real;
  delete[] p_imag;
  delete[] h_real;
  delete[] h_imag;
}

int main(int argc, char** argv) {
  int width = atoi(argv[1]);   // matrix width
  int height = atoi(argv[2]);  // matrix height
  int repeat = atoi(argv[3]);  // repeat kernel execution

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  tsa<float>(q, width, height, repeat);
  tsa<double>(q, width, height, repeat);
  return 0;
}
