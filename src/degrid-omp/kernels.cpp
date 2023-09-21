#include <chrono>

template <typename CmplxType>
void
degrid_kernel(CmplxType* __restrict out, 
              const CmplxType* __restrict in, 
              const size_t npts,
              const CmplxType* __restrict img, 
              const size_t img_dim,
              const CmplxType* __restrict gcf)
{

  #pragma omp target teams distribute num_teams(NPOINTS/32) thread_limit(256)
  for(size_t n=0; n<NPOINTS; n++) {
    int sub_x = floorf(GCF_GRID*(in[n].x-floorf(in[n].x)));
    int sub_y = floorf(GCF_GRID*(in[n].y-floorf(in[n].y)));
    int main_x = floor(in[n].x); 
    int main_y = floor(in[n].y); 
    PRECISION sum_r = 0.0;
    PRECISION sum_i = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:sum_r, sum_i)
    for (int a=-GCF_DIM/2; a<GCF_DIM/2 ;a++)
      for (int b=-GCF_DIM/2; b<GCF_DIM/2 ;b++) {
        PRECISION r1 = img[main_x+a+IMG_SIZE*(main_y+b)].x; 
        PRECISION i1 = img[main_x+a+IMG_SIZE*(main_y+b)].y; 
        PRECISION r2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].x;
        PRECISION i2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].y;
        if (main_x+a >= 0 && main_y+b >= 0 && 
            main_x+a < IMG_SIZE && main_y+b < IMG_SIZE) {
          sum_r += r1*r2 - i1*i2; 
          sum_i += r1*i2 + r2*i1;
        }
      }
    out[n].x = sum_r;
    out[n].y = sum_i;
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

  #pragma omp target data map(to:img[0:IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM], \
                                 gcf[0:64*GCF_DIM*GCF_DIM],\
                                 in[0:NPOINTS]) \
                          map(from:out[0:NPOINTS])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < REPEAT; n++) {
      // GCF_DIM is at least 32
      degrid_kernel(out, in, NPOINTS,
                    img + IMG_SIZE*GCF_DIM+GCF_DIM,
                    IMG_SIZE,
                    gcf + GCF_DIM*(GCF_DIM+1)/2);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / REPEAT << " (s)\n";
  }
}
