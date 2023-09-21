#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>

void zoom_in_kernel(
    sycl::nd_item<3> &item,
    float *__restrict staging_tile,
    const float *input_tensor,float *__restrict output_tensor,
    int input_h, int input_w, int output_h, int output_w,
    int pitch, int out_h_start, int out_h_end,
    int out_w_start, int out_w_end)
{
  // H -> block Y, row
  // W -> block X, col
  int out_start_h = item.get_group(1) * item.get_local_range(1);
  int out_end_h = (item.get_group(1) + 1) * item.get_local_range(1) - 1;
  int out_start_w = item.get_group(2) * item.get_local_range(2);
  int out_end_w = (item.get_group(2) + 1) * item.get_local_range(2) - 1;

  int img_start_offset = item.get_group(0) * pitch;

  // ideally should go in unified register
  int smem_load_h_start = sycl::floor((out_start_h * input_h) / (float)output_h);
  int smem_load_h_end = sycl::ceil(((out_end_h + 1) * input_h) / (float)output_h);
  int smem_h_load_stretch = smem_load_h_end - smem_load_h_start;

  int smem_load_w_start = sycl::floor((out_start_w * input_w) / (float)output_w);
  int smem_load_w_end = sycl::ceil(((out_end_w + 1) * input_w) / (float)output_w);
  int smem_w_load_stretch = smem_load_w_end - smem_load_w_start;

  for (int i = item.get_local_id(1); i < smem_h_load_stretch;
       i += item.get_local_range(1)) {
    for (int j = item.get_local_id(2); j < smem_w_load_stretch;
         j += item.get_local_range(2)) {

      if (((i+smem_load_h_start) < input_h) &&
          ((j+smem_load_w_start) < input_w)) {
        staging_tile[i * smem_w_load_stretch + j] = \
                      input_tensor[img_start_offset +
                      (smem_load_h_start + i) * input_w +
                      smem_load_w_start + j];
      } else {
        staging_tile[i * smem_w_load_stretch + j] = 0.0f;
      }
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  int out_pixel_h = item.get_global_id(1);
  int out_pixel_w = item.get_global_id(2);

  if (out_pixel_h < output_h && out_pixel_w < output_w
      && out_pixel_h >= out_h_start && out_pixel_h < out_h_end
      && out_pixel_w >= out_w_start && out_pixel_w < out_w_end) {

    // compute pixels oh, ow span
    int start_h = sycl::floor((out_pixel_h * input_h) / (float)output_h);
    int end_h = sycl::ceil(((out_pixel_h + 1) * input_h) / (float)output_h);

    int start_w = sycl::floor((out_pixel_w * input_w) / (float)output_w);
    int end_w = sycl::ceil(((out_pixel_w + 1) * input_w) / (float)output_w);

    int del_h = end_h - start_h;
    int del_w = end_w - start_w;

    float sum_ = 0.0f;

    for (int i = 0; i < del_h; i++) {
      for (int j = 0; j < del_w; j++) {
        int smem_row = (start_h + i) - smem_load_h_start;
        int smem_col = (start_w + j) - smem_load_w_start;
        sum_ += staging_tile[smem_row * smem_w_load_stretch + smem_col];
      }
    }
    sum_ /= (float)del_h;
    sum_ /= (float)del_w;

    output_tensor[(item.get_group(0) * pitch) +
                  ((out_pixel_h - out_h_start) * input_w) +
                  (out_pixel_w - out_w_start)] = sum_;
  }
}


void zoom_out_kernel(
    sycl::nd_item<3> &item,
    float *__restrict staging_tile,
    const float *input_tensor, float *__restrict output_tensor,
    int input_h, int input_w, int output_h, int output_w,
    int pitch, int out_h_start, int out_h_end,
    int out_w_start, int out_w_end)
{
  // H -> block Y, row
  // W -> block X, col
  int out_start_h = item.get_group(1) * item.get_local_range(1);
  int out_end_h = (item.get_group(1) + 1) * item.get_local_range(1) - 1;
  int out_start_w = item.get_group(2) * item.get_local_range(2);
  int out_end_w = (item.get_group(2) + 1) * item.get_local_range(2) - 1;

  int img_start_offset = item.get_group(0) * pitch;

  // ideally should go in unified register
  int smem_load_h_start = sycl::floor((out_start_h * input_h) / (float)output_h);
  int smem_load_h_end = sycl::ceil(((out_end_h + 1) * input_h) / (float)output_h);
  int smem_h_load_stretch = smem_load_h_end - smem_load_h_start;

  int smem_load_w_start = sycl::floor((out_start_w * input_w) / (float)output_w);
  int smem_load_w_end = sycl::ceil(((out_end_w + 1) * input_w) / (float)output_w);
  int smem_w_load_stretch = smem_load_w_end - smem_load_w_start;

  for (int i = item.get_local_id(1); i < smem_h_load_stretch;
       i += item.get_local_range(1)) {
    for (int j = item.get_local_id(2); j < smem_w_load_stretch;
         j += item.get_local_range(2)) {

      if (((i+smem_load_h_start) < input_h) &&
          ((j+smem_load_w_start) < input_w)) {
        staging_tile[i * smem_w_load_stretch + j] = \
                      input_tensor[img_start_offset +
                      (smem_load_h_start + i)*input_w +
                      smem_load_w_start + j];
      } else {
        staging_tile[i * smem_w_load_stretch + j] = 0.0f;
      }
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  int out_pixel_h = item.get_global_id(1);
  int out_pixel_w = item.get_global_id(2);

  if (out_pixel_h < output_h && out_pixel_w < output_w) {

    // compute pixels oh, ow span
    int start_h = sycl::floor((out_pixel_h * input_h) / (float)output_h);
    int end_h = sycl::ceil(((out_pixel_h + 1) * input_h) / (float)output_h);

    int start_w = sycl::floor((out_pixel_w * input_w) / (float)output_w);
    int end_w = sycl::ceil(((out_pixel_w + 1) * input_w) / (float)output_w);

    int del_h = end_h - start_h;
    int del_w = end_w - start_w;

    float sum_ = 0.0f;

    for (int i = 0; i < del_h; i++) {
      for (int j = 0; j < del_w; j++) {
        int smem_row = (start_h + i) - smem_load_h_start;
        int smem_col = (start_w + j) - smem_load_w_start;
        sum_ += staging_tile[smem_row * smem_w_load_stretch + smem_col];
      }
    }
    sum_ /= (float)del_h;
    sum_ /= (float)del_w;

    output_tensor[(item.get_group(0) * pitch) +
                  ((out_pixel_h + out_h_start) * input_w) +
                  (out_pixel_w + out_w_start)] = sum_;
  }
}


void zoom_out_edge_pad(
    sycl::nd_item<3> &item,
    float *output_tensor,
    int height, int width,
    int pitch, int no_padding_h_start,
    int no_padding_w_start,
    int no_padding_h_end, int no_padding_w_end)
{
  // H -> block Y, row
  // W -> block X, col

  int out_pixel_h = item.get_global_id(1);
  int out_pixel_w = item.get_global_id(2);

  // no_padding_h_end, no_padding_w_end --> w_cropped+wstart, same for height
  int out_location =
      (item.get_group(0) * pitch) + (out_pixel_h * width) + out_pixel_w;

  if (out_pixel_h < height && out_pixel_w < width) {
    if (out_pixel_h < no_padding_h_start && out_pixel_w >= no_padding_w_start
        && out_pixel_w < no_padding_w_end) {
      // top pad
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        (no_padding_h_start * width) + out_pixel_w];
    } else if (out_pixel_h >= no_padding_h_end
        && out_pixel_w >= no_padding_w_start
        && out_pixel_w < no_padding_w_end) {
      // bottom pad
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        ((no_padding_h_end - 1) * width) + out_pixel_w];
    } else if (out_pixel_w < no_padding_w_start
        && out_pixel_h >= no_padding_h_start
        && out_pixel_h < no_padding_h_end) {
      // left pad
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        (out_pixel_h * width) + no_padding_w_start];
    } else if (out_pixel_w >= no_padding_w_end
        && out_pixel_h >= no_padding_h_start
        && out_pixel_h < no_padding_h_end) {
      // right pad
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        (out_pixel_h * width) + (no_padding_w_end - 1)];
    } else if (out_pixel_h < no_padding_h_start
        && out_pixel_w < no_padding_w_start) {
      // top-left corner
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        (no_padding_h_start * width) + no_padding_w_start];
    } else if (out_pixel_h < no_padding_h_start
        && out_pixel_w >= no_padding_w_end) {
      // top-right corner
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        (no_padding_h_start * width) + (no_padding_w_end - 1)];
    } else if (out_pixel_h >= no_padding_h_end
        && out_pixel_w < no_padding_w_start) {
      // bottom-left corner
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        ((no_padding_h_end - 1) * width) + no_padding_w_start];
    } else if (out_pixel_h >= no_padding_h_end
        && out_pixel_w >= no_padding_w_end) {
      // bottom-right corner
      output_tensor[out_location] =
          output_tensor[(item.get_group(0) * pitch) +
                        ((no_padding_h_end - 1) * width) +
                        (no_padding_w_end - 1)];
    }
  }
}

int get_sm_size(int *output_sizes, int H, int W, sycl::range<3> &block) {

  int max_smem = 48 * 1024;
  int blocks[4][3] = {{16, 16, 1}, {16, 8, 1}, {8, 8, 1}, {8, 4, 1}};
  for (int i = 0; i < 4; i++) {
    int *param = blocks[0];
    int h_stretch = ceil((param[1] * H) / (float)output_sizes[2]);
    int w_stretch = ceil((param[0] * W) / (float)output_sizes[3]);
    int smem_size = (h_stretch + 1) * (w_stretch + 1) * 4;
    if (smem_size < max_smem) {
      block[2] = param[0];
      block[1] = param[1];
      block[0] = param[2];
      return smem_size / 4;
    }
  }
  printf("Requested shared memory size exceeds the maximum (%d). Exit\n", max_smem);
  exit(1);
}

void zoom (sycl::queue &q, int repeat, int input_sizes[4], float zoom_factor[2])
{
  int N = input_sizes[0];
  int C = input_sizes[1];
  int H = input_sizes[2];
  int W = input_sizes[3];

  // {N, C, H, W}
  int output_sizes[] = {N, C, (int)floor(H * zoom_factor[0]), (int)floor(W * zoom_factor[1])};

  bool is_zoom_out = output_sizes[2] < H && output_sizes[3] < W;
  bool is_zoom_in = output_sizes[2] > H && output_sizes[3] > W;
  if (is_zoom_out == false && is_zoom_in == false) {
    printf("Zoom factors only handle simultaneous expansion(or shrinkage) in both dimensions. Exit\n");
    exit(1);
  }

  // input pitch
  int pitch = H * W;

  // get block size and shared memory size
  sycl::range<3> block (1, 1, 1);
  int smem_size = get_sm_size(output_sizes, H, W, block);

  sycl::range<3> grid (C * N,
                      int((output_sizes[2] - 1) / block[1] + 1),
                      int((output_sizes[3] - 1) / block[2] + 1));

  int pad_dims[2][2] = {{0, 0}, {0,0}};  // zoom out
  int slice_dims[2][2] = {{0, 0}, {0,0}};  // zoom in

  int diff = H - output_sizes[2];
  int half = abs(diff) / 2;
  if (diff > 0) {
    pad_dims[0][0] = half;
    pad_dims[0][1] = diff - half;
  } else {
    slice_dims[0][0] = half;
    slice_dims[0][1] = H + half;
  }

  diff = W - output_sizes[3];
  half = abs(diff) / 2;
  if (diff > 0) {
    pad_dims[1][0] = half;
    pad_dims[1][1] = diff - half;
  } else {
    slice_dims[1][0] = half;
    slice_dims[1][1] = W + half;
  }

  size_t img_size = N * C * H * W;
  size_t img_size_bytes = sizeof(float) * img_size;

  float *input_img = (float*) malloc (img_size_bytes);
  float *output_img = (float*) malloc (img_size_bytes);

  float *d_input_img = (float *)sycl::malloc_device(img_size_bytes, q);
  float *d_output_img = (float *)sycl::malloc_device(img_size_bytes, q);

  std::default_random_engine rng (123);
  std::normal_distribution<float> norm_dist(0.f, 1.f);

  for (size_t i = 0; i < img_size; i++) {
    input_img[i] = norm_dist(rng);
  }

  q.memcpy(d_input_img, input_img, img_size_bytes);

  long total_time = 0;
  for (int i = 0; i < repeat; i++) {

    q.memset(d_output_img, 0, img_size_bytes);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    if (is_zoom_in) {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sm (sycl::range<1>(smem_size), cgh);

        auto o_h = output_sizes[2];
        auto o_w = output_sizes[3];
        auto o_h_start = slice_dims[0][0];
        auto o_h_end = slice_dims[0][1];
        auto o_w_start = slice_dims[1][0];
        auto o_w_end = slice_dims[1][1];

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) {
          zoom_in_kernel(item, sm.get_pointer(), d_input_img, d_output_img, H, W,
                         o_h, o_w, pitch, o_h_start, o_h_end, o_w_start, o_w_end);
        });
      });
    }
    else if (is_zoom_out) {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sm (sycl::range<1>(smem_size), cgh);

        auto o_h = output_sizes[2];
        auto o_w = output_sizes[3];
        auto o_h_start = pad_dims[0][0];
        auto o_h_end = pad_dims[0][1];
        auto o_w_start = pad_dims[1][0];
        auto o_w_end = pad_dims[1][1];

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) {
          zoom_out_kernel(item, sm.get_pointer(), d_input_img, d_output_img, H, W,
                          o_h, o_w, pitch, o_h_start, o_h_end, o_w_start, o_w_end);
        });
      });

      sycl::range<3> grid2(C * N,
                           int((H - 1) / block[1] + 1),
                           int((W - 1) / block[2] + 1));

      q.submit([&](sycl::handler &cgh) {
        auto np_h_start = pad_dims[0][0];
        auto np_w_start = pad_dims[1][0];
        auto np_h_end = pad_dims[0][0] + output_sizes[2];
        auto np_w_end = pad_dims[1][0] + output_sizes[3];

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block, block), [=](sycl::nd_item<3> item) {
          zoom_out_edge_pad(item, d_output_img, H, W, pitch,
                            np_h_start, np_w_start,
                            np_h_end, np_w_end);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  q.memcpy(output_img, d_output_img, img_size_bytes).wait();

  double checksum = 0;
  for (size_t i = 0; i < img_size; i++) {
    checksum += output_img[i];
  }

  sycl::free(d_input_img, q);
  sycl::free(d_output_img, q);
  free(input_img);
  free(output_img);

  printf("Average execution time of the %s kernel: %f (us)\n",
         is_zoom_in ? "zoom-in" : "zoom-out", total_time * 1e-3 / repeat);
  printf("Kernel checksum: %lf\n", checksum);
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <batch> <channel> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_sizes[4];
  input_sizes[0] = atoi(argv[1]);
  input_sizes[1] = atoi(argv[2]);
  input_sizes[2] = atoi(argv[3]);
  input_sizes[3] = atoi(argv[4]);
  int repeat = atoi(argv[5]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float zf[2]; // zoom factor

  zf[0] = 1.5f; zf[1] = 2.5f;
  zoom(q, repeat, input_sizes, zf);

  zf[0] = 0.6f; zf[1] = 0.9f;
  zoom(q, repeat, input_sizes, zf);

  return 0;
}
