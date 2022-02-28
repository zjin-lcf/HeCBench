#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"

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
 
  // input image
  float *img = (float*) malloc (sizeof(float) * size);

  // host and device results
  float *norm = (float*) malloc (sizeof(float) * size);
  float *h_norm = (float*) malloc (sizeof(float) * size);

  int *box = (int*) malloc (sizeof(int) * size);
  int *h_box = (int*) malloc (sizeof(int) * size);

  float *out = (float*) malloc (sizeof(float) * size);
  float *h_out = (float*) malloc (sizeof(float) * size);

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = rand() % 256;
    norm[i] = box[i] = out[i] = 0;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float,1> d_img (size);
  buffer<float,1> d_norm (size);
  buffer<  int,1> d_box (size);
  buffer<float,1> d_out (out, size);
  d_out.set_final_data(nullptr);

  range<2> gws ((Ly+15)/16*16, (Lx+15)/16*16);
  range<2> lws (16, 16);

  for (int i = 0; i < repeat; i++) {
    // restore input image
    q.submit([&] (handler &cgh) {
      auto acc = d_img.get_access<sycl_discard_write>(cgh);
      cgh.copy(img, acc); 
    });

    // reset norm
    q.submit([&] (handler &cgh) {
      auto acc = d_norm.get_access<sycl_discard_write>(cgh);
      cgh.copy(norm, acc); 
    });

    // launch three kernels
    q.submit([&] (handler &cgh) {
      auto Img = d_img.get_access<sycl_read>(cgh);
      auto Box = d_box.get_access<sycl_discard_write>(cgh);
      auto Norm = d_norm.get_access<sycl_read_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> s_Img(1024, cgh);
      cgh.parallel_for<class smoothing>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
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
          s_Img[stid] = Img[gtid];

        item.barrier(access::fence_space::local_space);

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
                    sum += Img[gtid + ii*Lx + jj];
                }
            q++;
          }
          Box[gtid] = s;

          // Normalization for each box
          for (int ii = -s; ii < s+1; ii++)
            for (int jj = -s; jj < s+1; jj++)
              if (ksum != 0) {
                auto atomic_obj_ref = sycl::atomic_ref<float, 
                  sycl::memory_order::relaxed,
                  sycl::memory_scope::device,
                  access::address_space::global_space> (Norm[gtid + ii*Lx + jj]);
                atomic_obj_ref.fetch_add(sycl::native::divide(1.f, (float)ksum));
              }
        }
      });
    });

    q.submit([&] (handler &cgh) {
      auto Norm = d_norm.get_access<sycl_read>(cgh);
      auto Img = d_img.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class normalize>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        int i = item.get_global_id(1);
        int j = item.get_global_id(0); 
        if ( i < Lx && j < Ly ) {
          int gtid = j * Lx + i;  
          const float norm = Norm[gtid];
          if (norm != 0) Img[gtid] = sycl::native::divide(Img[gtid], norm);
        }
      });
    });

    q.submit([&] (handler &cgh) {
      auto Img = d_img.get_access<sycl_read>(cgh);
      auto Box = d_box.get_access<sycl_read>(cgh);
      auto Out = d_out.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> s_Img(1024, cgh);
      cgh.parallel_for<class output>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        int tid = item.get_local_id(1);
        int tjd = item.get_local_id(0);
        int i = item.get_global_id(1);
        int j = item.get_global_id(0);
        int blockDim_x = item.get_local_range(1);
        int blockDim_y = item.get_local_range(0);
        int stid = tjd * blockDim_x + tid;
        int gtid = j * Lx + i;  

        if ( i < Lx && j < Ly )
          s_Img[stid] = Img[gtid];

        item.barrier(access::fence_space::local_space);

        if ( i < Lx && j < Ly )
        {
          const int s = Box[gtid];
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
                  sum += Img[gtid + ii*Ly + jj];
              }
          if ( ksum != 0 ) Out[gtid] = sycl::native::divide(sum, (float)ksum);
        }
      });
    });
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_out.get_access<sycl_read>(cgh);
    cgh.copy(acc, out);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_box.get_access<sycl_read>(cgh);
    cgh.copy(acc, box);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_norm.get_access<sycl_read>(cgh);
    cgh.copy(acc, norm); 
  });

  q.wait();

  // verify
  reference (Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  free(img);
  free(norm);
  free(h_norm);
  free(box);
  free(h_box);
  free(out);
  free(h_out);
  return 0;
}
