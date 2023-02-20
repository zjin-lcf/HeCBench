#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>

#define threadsPerBlock 512

// Kernel for fast unfold+copy on volumes
template <typename T>
__global__ void vol2col_kernel(
    const size_t range,
    const T* data_vol,
    const int depth,
    const int height,
    const int width,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    T* data_col)
{
  for (int64_t n = blockDim.x * blockIdx.x + threadIdx.x;
               n < range; n += blockDim.x * gridDim.x) {

    int w_out = n % width_col;
    int64_t index = n / width_col;
    int h_out = index % height_col;
    index = index / height_col;
    int t_out = index % depth_col;
    int channel_in = index / depth_col;
    int channel_out = channel_in * ksize_t * ksize_h * ksize_w;
    int t_in = t_out * stride_t - pad_t;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_vol += ((channel_in * depth + t_in) * height + h_in) * width + w_in;
    data_col += ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out;

    for (int i = 0; i < ksize_t; ++i) {
      for (int j = 0; j < ksize_h; ++j) {
        for (int k = 0; k < ksize_w; ++k) {
          int t = t_in + i * dilation_t;
          int h = h_in + j * dilation_h;
          int w = w_in + k * dilation_w;
          *data_col = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width)
              ? data_vol[i * dilation_t * height * width +
                         j * dilation_h * width + k * dilation_w]
              : static_cast<T>(0);
          data_col += depth_col * height_col * width_col;
        }
      }
    }
  }
}

int get_blocks (size_t n) {
  size_t numBlocks = (n - threadsPerBlock) / threadsPerBlock + 1;
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  size_t blocksPerGrid = min((size_t)devProp.maxGridSize[0], numBlocks);
  return blocksPerGrid;
}

template <typename T>
void eval (
    const int repeat,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int depth_col,
    const int height_col,
    const int width_col,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w)
{
  size_t vol_size = (size_t) channels * (2*pad_t+depth) * (2*pad_h+height) * (2*pad_w+width);
  size_t col_size = ((size_t) channels * ksize_t * ksize_h * ksize_w + 1) * 
                    (depth_col+pad_t) * (height_col+pad_h) * (width_col+pad_w);
                         
  size_t vol_size_bytes = sizeof(T) * vol_size;
  size_t col_size_bytes = sizeof(T) * col_size;
  
  T *h_data_vol = (T*) malloc (vol_size_bytes);
  T *h_data_col = (T*) malloc (col_size_bytes);

  for (size_t i = 0; i < vol_size; i++) {
    h_data_vol[i] = (T)1; 
  }

  T *d_data_vol;
  hipMalloc((void**)&d_data_vol, vol_size_bytes);
  hipMemcpy(d_data_vol, h_data_vol, vol_size_bytes, hipMemcpyHostToDevice);

  T *d_data_col;
  hipMalloc((void**)&d_data_col, col_size_bytes);
  hipMemset(d_data_col, 0, col_size_bytes);

  // each of "channels * depth_col * height_col * width_col"
  // blocks responsible for copying a single-channel grid.
  // We cast an operand to int64 so that the product will not overflow
  size_t n = static_cast<size_t>(channels) * depth_col * height_col * width_col;

  int blocksPerGrid = get_blocks(n);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    vol2col_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(
      n,
      d_data_vol,
      depth, height, width,
      ksize_t, ksize_h, ksize_w,
      pad_t, pad_h, pad_w,
      stride_t, stride_h, stride_w,
      dilation_t, dilation_h, dilation_w,
      depth_col, height_col, width_col,
      d_data_col);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of vol2col kernel: %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(h_data_col, d_data_col, col_size_bytes, hipMemcpyDeviceToHost);

  float checksum = 0;
  for (size_t i = 0; i < col_size; i++) {
    checksum += h_data_col[i];
  }
  printf("Checksum = %f\n", checksum / col_size);

  hipFree(d_data_vol);
  hipFree(d_data_col);
  free(h_data_vol);
  free(h_data_col);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  
  int channels = 4;
  int depth = 3;
  int height = 255;
  int width = 255;
  int pad_t = 1;
  int pad_h = 1;
  int pad_w = 1;
  int stride_t = 2;
  int stride_h = 2;
  int stride_w = 2;
  int dilation_t = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  int depth_col = 3;
  int height_col = 255;
  int width_col = 255;

  for (int k = 1; k <= 9; k = k + 2) {
    printf("\nkernel size: %d\n", k);
    int ksize_t = k;
    int ksize_h = k;
    int ksize_w = k;

    eval<float> (repeat,
                 channels, depth, height, width,
                 depth_col, height_col, width_col,
                 ksize_t, ksize_h, ksize_w,
                 pad_t, pad_h, pad_w,
                 stride_t, stride_h, stride_w,
                 dilation_t, dilation_h, dilation_w);
  }

  return 0;
}
