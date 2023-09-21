#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

#include "reference.cpp"

int main(int argc, char* argv[]) {
  if (argc != 5) {
     printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
     exit(1);
  }

  // only a square image is supported
  const int Lx = atoi(argv[1]);
  const int Ly = Lx;
  const int size = Lx * Ly;

  const int Threshold = atoi(argv[2]);
  const int MaxRad = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const size_t size_bytes = size * sizeof(float);
  const size_t box_bytes = size * sizeof(int);
 
  // input image
  float *img = (float*) malloc (size_bytes);

  // host and device results
  float *norm = (float*) malloc (size_bytes);
  float *h_norm = (float*) malloc (size_bytes);

  int *box = (int*) malloc (box_bytes);
  int *h_box = (int*) malloc (box_bytes);

  float *out = (float*) malloc (size_bytes);
  float *h_out = (float*) malloc (size_bytes);

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = rand() % 256;
    norm[i] = box[i] = out[i] = 0;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_img = sycl::malloc_device<float>(size, q);
  float *d_norm = sycl::malloc_device<float>(size, q);
    int *d_box = sycl::malloc_device<int>(size, q);
  float *d_out = sycl::malloc_device<float>(size, q);

  sycl::range<2> gws ((Ly+15)/16*16, (Lx+15)/16*16);
  sycl::range<2> lws (16, 16);

  double time = 0;

  for (int i = 0; i < repeat; i++) {
    // restore input image
    q.memcpy(d_img, img, size_bytes);

    // reset norm
    q.memcpy(d_norm, norm, size_bytes);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    // launch three kernels
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> s_Img(sycl::range<1>(1024), cgh);
      cgh.parallel_for<class smoothing>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        int tid = item.get_local_id(1);
        int tjd = item.get_local_id(0);
        int i = item.get_global_id(1);
        int j = item.get_global_id(0);
        int blockDim_x = item.get_local_range(1);
        int blockDim_y = item.get_local_range(0);
        int stid = tjd * blockDim_x + tid;
        int gtid = j * Lx + i;  

        // part of shared memory may be unused

        if ( i < Lx && j < Ly )
          s_Img[stid] = d_img[gtid];

        item.barrier(sycl::access::fence_space::local_space);

        if ( i < Lx && j < Ly )
        {
          // Smoothing parameters
          float sum = 0.f;
          int q = 1;
          int s = q;
          int ksum = 0;

          // Continue until parameters are met
          while (sum < Threshold && q < MaxRad)
          {
            s = q;
            sum = 0.f;
            ksum = 0;

            // Normal adaptive smoothing
            for (int ii = -s; ii < s+1; ii++)
              for (int jj = -s; jj < s+1; jj++)
                if ( (i-s >= 0) && (i+s < Ly) && (j-s >= 0) && (j+s < Lx) )
                {
                  ksum++;
                  // Compute within bounds of block dimensions
                  if( tid-s >= 0 && tid+s < blockDim_x && tjd-s >= 0 && tjd+s < blockDim_y )
                    sum += s_Img[stid + ii*blockDim_x + jj];
                  // Compute block borders with global memory
                  else
                    sum += d_img[gtid + ii*Lx + jj];
                }
            q++;
          }
          d_box[gtid] = s;

          // Normalization for each box
          for (int ii = -s; ii < s+1; ii++)
            for (int jj = -s; jj < s+1; jj++)
              if (ksum != 0) {
                auto ao = sycl::atomic_ref<float, 
                  sycl::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space> (d_norm[gtid + ii*Lx + jj]);
                ao.fetch_add(sycl::native::divide(1.f, (float)ksum));
              }
        }
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class normalize>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        int i = item.get_global_id(1);
        int j = item.get_global_id(0); 
        if ( i < Lx && j < Ly ) {
          int gtid = j * Lx + i;  
          const float norm = d_norm[gtid];
          if (norm != 0) d_img[gtid] = sycl::native::divide(d_img[gtid], norm);
        }
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> s_Img(1024, cgh);
      cgh.parallel_for<class output>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        int tid = item.get_local_id(1);
        int tjd = item.get_local_id(0);
        int i = item.get_global_id(1);
        int j = item.get_global_id(0);
        int blockDim_x = item.get_local_range(1);
        int blockDim_y = item.get_local_range(0);
        int stid = tjd * blockDim_x + tid;
        int gtid = j * Lx + i;  

        if ( i < Lx && j < Ly )
          s_Img[stid] = d_img[gtid];

        item.barrier(sycl::access::fence_space::local_space);

        if ( i < Lx && j < Ly )
        {
          const int s = d_box[gtid];
          float sum = 0.f;
          int ksum  = 0;

          for (int ii = -s; ii < s+1; ii++)
            for (int jj = -s; jj < s+1; jj++)
              if ( (i-s >= 0) && (i+s < Lx) && (j-s >= 0) && (j+s < Ly) )
              {
                ksum++;
                if( tid-s >= 0 && tid+s < blockDim_x && tjd-s >= 0 && tjd+s < blockDim_y )
                  sum += s_Img[stid + ii*blockDim_y + jj];
                else
                  sum += d_img[gtid + ii*Ly + jj];
              }
          if ( ksum != 0 ) d_out[gtid] = sycl::native::divide(sum, (float)ksum);
        }
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average filtering time %lf (s)\n", (time * 1e-9) / repeat);

  q.memcpy(out, d_out, size_bytes);
  q.memcpy(box, d_box, box_bytes);
  q.memcpy(norm, d_norm, size_bytes);

  q.wait();

  // verify
  reference (Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  sycl::free(d_img, q);
  sycl::free(d_norm, q);
  sycl::free(d_box, q);
  sycl::free(d_out, q);
  free(img);
  free(norm);
  free(h_norm);
  free(box);
  free(h_box);
  free(out);
  free(h_out);
  return 0;
}
