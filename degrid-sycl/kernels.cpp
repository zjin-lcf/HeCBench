#include <chrono>

// Forward declarations
template <typename CmplxType>
class degrid;

template <typename CmplxType>
void
degrid_kernel(CmplxType *out, 
              const CmplxType *in, 
              const size_t npts,
              const CmplxType *img, 
              const size_t img_dim,
              const CmplxType *gcf,
              sycl::nd_item<2> &item)
{
  const int blockIdx_x = item.get_group(1);
  const int blockDim_x = item.get_local_range(1);
  const int threadIdx_x = item.get_local_id(1);
  const int gridDim_x = item.get_group_range(1);
  const int blockDim_y = item.get_local_range(0);
  const int threadIdx_y = item.get_local_id(0);

  for (int n = 32*blockIdx_x; n < npts; n += 32*gridDim_x) {
    for (int q = threadIdx_y; q < 32; q += blockDim_y) {
      CmplxType inn = in[n+q];
      const int sub_x = sycl::floor(GCF_GRID*(inn.x()-sycl::floor(inn.x())));
      const int sub_y = sycl::floor(GCF_GRID*(inn.y()-sycl::floor(inn.y())));
      const int main_x = sycl::floor(inn.x()); 
      const int main_y = sycl::floor(inn.y()); 
      CmplxType sum = {0,0};
      for(int a = threadIdx_x-GCF_DIM/2; a < GCF_DIM/2; a += blockDim_x)
        for(int b = -GCF_DIM/2; b < GCF_DIM/2; b++)
        {
          auto r1 = img[main_x+a+img_dim*(main_y+b)].x(); 
          auto i1 = img[main_x+a+img_dim*(main_y+b)].y(); 
          if (main_x+a < 0 || main_y+b < 0 || 
              main_x+a >= img_dim  || main_y+b >= img_dim) {
            r1 = i1 = 0;
          }
          auto r2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].x();
          auto i2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].y();
          sum.x() += r1*r2 - i1*i2; 
          sum.y() += r1*i2 + r2*i1;
        }

      for(int s = blockDim_x < 16 ? blockDim_x : 16; s>0;s/=2) {
        sum.x() += item.get_sub_group().shuffle_down(sum.x(),s);
        sum.y() += item.get_sub_group().shuffle_down(sum.y(),s);
      }
      if (threadIdx_x == 0) {
        out[n+q] = sum;
      }
    }
  }
}

template <typename CmplxType>
void degridGPU(CmplxType* out, CmplxType* in, CmplxType *img, CmplxType *gcf) {
  //degrid on the GPU
  //  out (out) - the output values for each location
  //  in  (in)  - the locations to be interpolated 
  //  img (in) - the image
  //  gcf (in) - the gridding convolution function

  //img is padded to avoid overruns. Subtract to find the real head
  img -= IMG_SIZE*GCF_DIM+GCF_DIM;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const size_t image_size = IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM;
  const size_t image_size_bytes = image_size * sizeof(CmplxType);
 
  CmplxType *d_img = sycl::malloc_device<CmplxType> (image_size, q); 
  CmplxType *d_gcf = sycl::malloc_device<CmplxType> (64*GCF_DIM*GCF_DIM, q);
  CmplxType *d_out = sycl::malloc_device<CmplxType> (NPOINTS, q);
  CmplxType *d_in  = sycl::malloc_device<CmplxType> (NPOINTS, q);

  q.memcpy(d_img, img, image_size_bytes);
  q.memcpy(d_in, in, sizeof(CmplxType) * NPOINTS);
  q.memcpy(d_gcf, gcf, sizeof(CmplxType) * 64*GCF_DIM*GCF_DIM);

  // NPOINTS is a multiple of 32
  sycl::range<2> gws(8, NPOINTS);
  sycl::range<2> lws(8, 32);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < REPEAT; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class degrid<CmplxType>>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        // GCF_DIM is at least 32
        degrid_kernel(d_out,
                      d_in,
                      NPOINTS,
                      d_img + IMG_SIZE*GCF_DIM+GCF_DIM,
                      IMG_SIZE,
                      d_gcf + GCF_DIM*(GCF_DIM+1)/2,
                      item); 
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / REPEAT << " (s)\n";

  q.memcpy(out, d_out, sizeof(CmplxType) * NPOINTS).wait();
  sycl::free(d_img, q);
  sycl::free(d_gcf, q);
  sycl::free(d_out, q);
  sycl::free(d_in, q);
}
