#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"

#define GPU_THREADS 256

#define KERNEL_LOOP(index, range) \
  for (int index = item.get_global_id(0);  \
           index < (range); \
           index += item.get_local_range(0) * item.get_group_range(0))

template <typename T>
void sequenceMaskKernel(
    nd_item<1> &item,
    int N,
    int M,
    int B,
    const T* in,
    const int* seq_lengths,
    T fill_val,
    T* out)
{
  if (B >= 0) {
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      out[ind] = (k >= seq_lengths[j] ? fill_val : in[ind]);
    }
  } else {
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;
      out[index] = (j >= seq_lengths[i] ? fill_val : in[index]);
    }
  }
}

template <typename T>
void windowMaskKernel(
    nd_item<1> &item,
    int N,
    int M,
    int B,
    const T* in,
    const int* window_centers,
    const int radius,
    T fill_val,
    T* out) {
  if (B >= 0) {
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] =
          (k < window_centers[j] - radius || k > window_centers[j] + radius
               ? fill_val
               : in[ind]);
    }
  } else {
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] =
          (j < window_centers[i] - radius || j > window_centers[i] + radius
               ? fill_val
               : in[index]);
    }
  }
}

template <typename T>
void upperMaskKernel(nd_item<1> &item, int N, int M, int B,
                     const T* in, T fill_val, T* out) 
{
  if (B >= 0) {
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k > j ? fill_val : in[ind]);
    }
  } else {
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j > i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void lowerMaskKernel(nd_item<1> &item, int N, int M, int B,
                     const T* in, T fill_val, T* out)
{
  if (B >= 0) {
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k < j ? fill_val : in[ind]);
    }
  } else {
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j < i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void upperDiagMaskKernel(nd_item<1> &item, int N, int M, int B,
                         const T* in, T fill_val, T* out)
{
  if (B >= 0) {
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k >= j ? fill_val : in[ind]);
    }
  } else {
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j >= i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void lowerDiagMaskKernel(nd_item<1> &item, int N, int M, int B,
                         const T* in, T fill_val, T* out)
{
  if (B >= 0) {
    KERNEL_LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k <= j ? fill_val : in[ind]);
    }
  } else {
    KERNEL_LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j <= i ? fill_val : in[index]);
    }
  }
}

template<typename T>
void print_mask_ratio (queue &q, T *h_out, T *d_out, T fill_val, int data_size) {
  q.memcpy(h_out, d_out, data_size * sizeof(T));
  int cnt_fill = 0;
  for (int i = 0; i < data_size; i++) {
    if (h_out[i] == fill_val) cnt_fill++;
  }
  printf("Mask ratio: %f\n", (float) cnt_fill / data_size);
}

template<typename T>
void eval_mask (const int M, const int N, const int B, const int repeat) {

  const T fill_val = -1;
  const int radius = M / 4;  // closely related to mask ratio

  int batch_dim = (B <= 0) ? 1 : B;

  printf("\nM = %d, N = %d, B = %d\n", M, N, batch_dim);

  int data_size = N * M * batch_dim;
  size_t data_size_in_bytes = data_size * sizeof(T); 

  int window_size = N;
  size_t window_size_in_bytes = N * sizeof(int);

  int seq_len = N;
  size_t seq_len_in_bytes = seq_len * sizeof(int); 

  T *h_in = (T*) malloc (data_size_in_bytes);
  T *h_out = (T*) malloc (data_size_in_bytes);
  int *h_seq_len = (int*) malloc (seq_len_in_bytes);
  int *h_window = (int*) malloc (window_size_in_bytes);

  srand(123);
  for (int i = 0; i < seq_len; i++) {
    h_seq_len[i] = rand() % (M / 2); // closely related to mask ratio
  }
  for (int i = 0; i < window_size; i++) {
    h_window[i] = rand() % M;
  }
  for (int i = 0; i < data_size; i++) {
    h_in[i] = rand() % (M * N);
  }
  
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  T *d_in, *d_out;
  int *d_seq_len, *d_window;
  d_in = malloc_device<T>(data_size, q);
  d_out = malloc_device<T>(data_size, q);
  d_window = malloc_device<int>(window_size, q);
  d_seq_len = malloc_device<int>(seq_len, q);

  q.memcpy(d_in, h_in, data_size_in_bytes);
  q.memcpy(d_seq_len, h_seq_len, seq_len_in_bytes);
  q.memcpy(d_window, h_window, window_size_in_bytes);

  int nblocks = (B <= 0) ? (N * M / GPU_THREADS) : (N * M);
  range<1> gws (nblocks * GPU_THREADS);
  range<1> lws (GPU_THREADS);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class sequence>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        sequenceMaskKernel(item, N, M, batch_dim, d_in, d_seq_len, fill_val, d_out);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of sequenceMask kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  print_mask_ratio(q, h_out, d_out, fill_val, data_size);
 
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class window>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        windowMaskKernel(item, N, M, batch_dim, d_in, d_window,
                         radius, fill_val, d_out);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of windowMask kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  print_mask_ratio(q, h_out, d_out, fill_val, data_size);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class upper>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        upperMaskKernel(item, N, M, batch_dim, d_in, fill_val, d_out);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of upperMask kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  print_mask_ratio(q, h_out, d_out, fill_val, data_size);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class lower>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        lowerMaskKernel(item, N, M, batch_dim, d_in, fill_val, d_out);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of lowerMask kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  print_mask_ratio(q, h_out, d_out, fill_val, data_size);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class upperDiag>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        upperDiagMaskKernel(item, N, M, batch_dim, d_in, fill_val, d_out);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of upperDiagMask kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  print_mask_ratio(q, h_out, d_out, fill_val, data_size);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class lowerDiag>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        lowerDiagMaskKernel(item, N, M, batch_dim, d_in, fill_val, d_out);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of lowerDiagMask kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  print_mask_ratio(q, h_out, d_out, fill_val, data_size);

  free(d_in, q);
  free(d_out, q);
  free(d_window, q);
  free(d_seq_len, q);

  free(h_in);
  free(h_out);
  free(h_window);
  free(h_seq_len);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <sequence length> <sequence length> <batch size> <repeat>\n", argv[0]);
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int B = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  eval_mask<int>(M, N, B, repeat);

  return 0;
}
