#include <chrono>

template <typename CmplxType>
__global__ void
degrid_kernel(CmplxType* __restrict out, 
              const CmplxType* __restrict in, 
              const size_t npts,
              const CmplxType* __restrict img, 
              const size_t img_dim,
              const CmplxType* __restrict gcf)
{
  const int blockIdx_x = blockIdx.x; 
  const int blockDim_x = blockDim.x; 
  const int threadIdx_x = threadIdx.x; 
  const int gridDim_x = gridDim.x; 
  const int blockDim_y = blockDim.y; 
  const int threadIdx_y = threadIdx.y;

  for (int n = 32*blockIdx_x; n < npts; n += 32*gridDim_x) {
    for (int q = threadIdx_y; q < 32; q += blockDim_y) {
      CmplxType inn = in[n+q];
      const int sub_x = floorf(GCF_GRID*(inn.x-floorf(inn.x)));
      const int sub_y = floorf(GCF_GRID*(inn.y-floorf(inn.y)));
      const int main_x = floorf(inn.x); 
      const int main_y = floorf(inn.y); 
      CmplxType sum = {0,0};
      for(int a = threadIdx_x-GCF_DIM/2; a < GCF_DIM/2; a += blockDim_x)
        for(int b = -GCF_DIM/2; b < GCF_DIM/2; b++)
        {
          auto r1 = img[main_x+a+img_dim*(main_y+b)].x; 
          auto i1 = img[main_x+a+img_dim*(main_y+b)].y; 
          if (main_x+a < 0 || main_y+b < 0 || 
              main_x+a >= img_dim  || main_y+b >= img_dim) {
            r1 = i1 = 0;
          }
          auto r2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].x;
          auto i2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].y;
          sum.x += r1*r2 - i1*i2; 
          sum.y += r1*i2 + r2*i1;
        }

      for(int s = blockDim_x < 16 ? blockDim_x : 16; s>0;s/=2) {
        sum.x += __shfl_down(sum.x,s);
        sum.y += __shfl_down(sum.y,s);
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

  CmplxType *d_img;
  hipMalloc((void**)&d_img, sizeof(CmplxType)*
             (IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM));

  hipMemcpy(d_img, img, sizeof(CmplxType)*
             (IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM), hipMemcpyHostToDevice);

  CmplxType *d_gcf;
  hipMalloc((void**)&d_gcf, sizeof(CmplxType)*64*GCF_DIM*GCF_DIM);
  hipMemcpy(d_gcf, gcf, sizeof(CmplxType)*64*GCF_DIM*GCF_DIM, hipMemcpyHostToDevice);

  CmplxType *d_out;
  hipMalloc((void**)&d_out, sizeof(CmplxType)*NPOINTS);

  CmplxType *d_in;
  hipMalloc((void**)&d_in, sizeof(CmplxType)*NPOINTS);
  hipMemcpy(d_in, in, sizeof(CmplxType)*NPOINTS, hipMemcpyHostToDevice);

  // NPOINTS is a multiple of 32
  dim3 grid(NPOINTS/32, 1);
  dim3 block(32, 8);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < REPEAT; n++) {
    // GCF_DIM is at least 32
    hipLaunchKernelGGL(degrid_kernel, grid, block, 0, 0, d_out, d_in, NPOINTS,
                  d_img + IMG_SIZE*GCF_DIM+GCF_DIM,
                  IMG_SIZE,
                  d_gcf + GCF_DIM*(GCF_DIM+1)/2);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / REPEAT << " (s)\n";

  hipMemcpy(out, d_out, sizeof(CmplxType)*NPOINTS, hipMemcpyDeviceToHost);
  hipFree(d_img);
  hipFree(d_in);
  hipFree(d_out);
  hipFree(d_gcf);
}
