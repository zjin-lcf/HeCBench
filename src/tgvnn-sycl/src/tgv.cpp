/*
  This file is part of the TGV package (https://github.com/chixindebaoyu/tgvnn).

  The MIT License (MIT)

  Copyright (c) Dong Wang

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to # deal
  in the Software without restriction, including without limitation the # rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or # sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING # FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS # IN THE
  SOFTWARE.
*/

#include <sycl/sycl.hpp>
#include "tgv.h"
#include <cmath>

//Configuration parameters
int blocksize = 128;
int gridsize = 128;

fft::fft_engine_ptr fft2_plan;

void *
mallocAssert(const size_t n, const char *file, int line)
{
    void *p = malloc(n);
    if (p == NULL)
    {
        printf("Malloc failed from %s:%d", file, line);
        abort();
    }
    return p;
}
#define safe_malloc(n) mallocAssert(n, __FILE__, __LINE__);

void
freeAssert(void *p, const char *file, int line)
{
    if (p == NULL)
    {
        printf("Free failed from %s:%d", file, line);
        abort();
    }
}
#define safe_free(x) freeAssert(x, __FILE__, __LINE__);

void
print_usage()
{
    fprintf(stderr, "Dynamic MRI reconstruction using Primal-Dual algorithm\n");
    fprintf(stderr, "Decomposition of second order TGV and nuclear norm\n");
    fprintf(stderr, "Usage: runtgv [OPTION] <img.ra> <mask.ra>\n");
    fprintf(stderr, "\t-o, --output <output.ra>\t recon output RA file\n");
    fprintf(stderr, "\t-i, --iter n\t\t\t iteration number\n");
    fprintf(stderr, "\t-a, --alpha n\t\t\t parameter for TGV\n");
    fprintf(stderr, "\t-b, --beta n\t\t\t parameter for nuclear norm\n");
    fprintf(stderr, "\t-s, --sigma n\t\t\t dual stepsize\n");
    fprintf(stderr, "\t-t, --tau n\t\t\t primal stepsize\n");
    fprintf(stderr, "\t-m, --mu n\t\t\t temporal stepsize\n");
    fprintf(stderr, "\t-G, --gridsize n\t\t set GPU gridsize\n");
    fprintf(stderr, "\t-B, --blocksize n\t\t set GPU blocksize\n");
    fprintf(stderr, "\t-h\t\t\t\t show this help\n");
}

void save_rafile(sycl::float2 *h_out, const char *out_path, const size_t dim0,
                 const size_t dim1, const size_t dim2, const size_t dim3)
{
    // save out
    ra_t out;
    out.flags = 0;
    out.eltype = RA_TYPE_COMPLEX;
    out.elbyte = sizeof(sycl::float2);
    out.size = dim0 * dim1 * dim2 * dim3 * sizeof(sycl::float2);
    out.ndims = 4;
    out.dims = (uint64_t *) safe_malloc(out.ndims * sizeof(uint64_t));
    out.dims[0] = dim0;
    out.dims[1] = dim1;
    out.dims[2] = dim2;
    out.dims[3] = dim3;
    out.data = (uint8_t *) h_out;
    ra_write(&out, out_path);
    ra_free(&out);
}

void scaledata(sycl::float2 *d_array, const size_t array_size,
               const float factor, const sycl::nd_item<3> &item)
{
    for (int id = item.get_group(2) * item.get_local_range(2) +
                  item.get_local_id(2);
         id < array_size;
         id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_array[id] *= factor;
    }
}

float
compute_alpha (float alpha0, float alpha1, int iter, int index)
{
  return expf(float(index)/float(iter)*logf(alpha1) +
          (float(iter)-float(index))/float(iter)*logf(alpha0));
}

float compute_maxmag(sycl::queue &q, sycl::float2 *d_array, const size_t array_size) {
  int max_idx;
  int *d_max_idx = sycl::malloc_device<int>(1, q);
  oneapi::mkl::blas::column_major::iamax(
      q, array_size, (std::complex<float> *)d_array, 1,
      d_max_idx, oneapi::mkl::index_base::one);
  q.memcpy(&max_idx, d_max_idx, sizeof(int)).wait();
  sycl::free(d_max_idx, q);

  sycl::float2 h_tmp;
  q.memcpy(&h_tmp, d_array + max_idx - 1, sizeof(sycl::float2)).wait();

  return abs(h_tmp.x());
}

void arrayadd(sycl::float2 *array_c, sycl::float2 *array_a,
              sycl::float2 *array_b, const size_t array_size, float alpha,
              float beta, const sycl::nd_item<3> &item)
{
    for (int id = item.get_group(2) * item.get_local_range(2) +
                  item.get_local_id(2);
         id < array_size;
         id += item.get_local_range(2) * item.get_group_range(2))
    {
      array_c[id] = alpha*array_a[id] + beta*array_b[id];
    }
}

void
arrayadd (float *array_c, float *array_a, float *array_b,
  const size_t array_size, float alpha, float beta,
  const sycl::nd_item<3> &item)
{
    for (int id = item.get_group(2) * item.get_local_range(2) +
                  item.get_local_id(2);
         id < array_size;
         id += item.get_local_range(2) * item.get_group_range(2))
    {
      array_c[id] = alpha*array_a[id] + beta*array_b[id];
    }
}

void arraydot(sycl::float2 *array_c, sycl::float2 *array_a,
              sycl::float2 *array_b, const size_t array_size,
              const sycl::nd_item<3> &item, float alpha = 1.f)
{
    for (int id = item.get_group(2) * item.get_local_range(2) +
                  item.get_local_id(2);
         id < array_size;
         id += item.get_local_range(2) * item.get_group_range(2))
    {
      array_c[id] = alpha*array_a[id]*array_b[id];
    }
}

void arrayreal(sycl::float2 *d_out, sycl::float2 *d_in, size_t array_size,
               const sycl::nd_item<3> &item)
{
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < array_size;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_out[id] = make_float2(sycl::fmax(d_in[id].x(), 0.f));
    }
}

void
fft2_init(sycl::queue &q, const int rows, const int cols, const int ndyn)
{
  // setup FFT plan, ndyn = # timepoints
  const int rank = 2;
  int idist = 1, odist = 1, istride = ndyn, ostride = ndyn;
  int n[rank] = {cols, rows};

  fft2_plan = fft::fft_engine::create(
      &q, rank, n, NULL, istride, idist, NULL, ostride,
      odist, fft::fft_type::complex_float_to_complex_float, ndyn);
}

void forward(sycl::queue &q, sycl::float2 *d_out, sycl::float2 *d_in, sycl::float2 *d_mask,
             const size_t N, const size_t rows, const size_t cols)
{
  fft2_plan->compute<sycl::float2, sycl::float2>(
      d_in, d_out, fft::fft_direction::forward);
  q.submit([&](sycl::handler &cgh) {
    auto sqrtf_rows_cols_ct2 = 1.f / sqrtf(rows * cols);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                           sycl::range<3>(1, 1, gridsize),
                                       sycl::range<3>(1, 1, gridsize)),
                     [=](sycl::nd_item<3> item) {
                       scaledata(d_out, N, sqrtf_rows_cols_ct2, item);
                     });
  });
  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        arraydot(d_out, d_out, d_mask, N, item, 1.f);
      });
}

void backward(sycl::queue &q, sycl::float2 *d_out, sycl::float2 *d_in, sycl::float2 *d_mask,
              const size_t N, const size_t rows, const size_t cols)
{
  sycl::float2 *d_tmp;
  d_tmp = sycl::malloc_device<sycl::float2>(N, q);

  q.memcpy(d_tmp, d_in, N * sizeof(sycl::float2));

  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        arraydot(d_tmp, d_tmp, d_mask, N, item, 1.f);
      });
  fft2_plan->compute<sycl::float2, sycl::float2>(
      d_tmp, d_out, fft::fft_direction::backward);
  q.submit([&](sycl::handler &cgh) {
    auto sqrtf_rows_cols_ct2 = 1.f / sqrtf(rows * cols);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                           sycl::range<3>(1, 1, gridsize),
                                       sycl::range<3>(1, 1, gridsize)),
                     [=](sycl::nd_item<3> item) {
                       scaledata(d_out, N, sqrtf_rows_cols_ct2, item);
                     });
  });

  sycl::free(d_tmp, q);
}

void update_r(sycl::float2 *d_r, sycl::float2 *d_tmp, sycl::float2 *d_imgb,
              size_t N, float sigma, const sycl::nd_item<3> &item)
{
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N; id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_tmp[id] = d_tmp[id] - d_imgb[id];
      d_r[id] = (d_r[id] + sigma*d_tmp[id])/(1.f+sigma);
    }
}

void grad_xx(sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
             const size_t rows, const size_t cols, const size_t ndyn,
             const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N - rows * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
  {
    // (the index inside a group) + (the index of the group) * offset
    idx = id%(rows*(cols-1)) + id/(rows*(cols-1))*(rows*cols);

    d_out[idx] = d_in[idx + rows] - d_in[idx];
  }
}

void grad_xx_bound(sycl::float2 *d_array, const size_t N, const size_t rows,
                   const size_t cols, const size_t ndyn,
                   const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < rows * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id%rows + id/rows*(rows*cols) + rows*(cols-1);

      d_array[idx] = make_float2(0.f);
    }
}

void grad_yy(sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
             const size_t rows, const size_t cols, const size_t ndyn,
             const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N - cols * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
  {
    idx = id%(rows-1) + id/(rows-1)*rows;

    d_out[idx] = d_in[idx + 1] - d_in[idx];
  }
}

void grad_yy_bound(sycl::float2 *d_array, const size_t N, const size_t rows,
                   const size_t cols, const size_t ndyn,
                   const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < cols * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id*rows + (rows-1);

      d_array[idx] = make_float2(0.f);
    }
}

void grad_tt(sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
             const size_t rows, const size_t cols, const size_t ndyn, float mu,
             const sycl::nd_item<3> &item)
{
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N - rows * cols;
       id += item.get_local_range(2) * item.get_group_range(2))
  {
    d_out[id] = (d_in[id + rows * cols] - d_in[id]) / mu;
  }
}

void grad_tt_bound(sycl::float2 *d_array, const size_t N, const size_t rows,
                   const size_t cols, const size_t ndyn,
                   const sycl::nd_item<3> &item)
{
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < rows * cols;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_array[id + (ndyn-1)*rows*cols] = make_float2(0.f);
    }
}

void update_r(sycl::queue &q, sycl::float2 *d_r, sycl::float2 *d_lbar, sycl::float2 *d_sbar,
              sycl::float2 *d_imgb, sycl::float2 *d_tmp, sycl::float2 *d_mask,
              float sigma, const size_t N, const size_t rows, const size_t cols)
{
  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        arrayadd(d_tmp, d_lbar, d_sbar, N, 1.f, 1.f, item);
      });
  forward(q, d_tmp, d_tmp, d_mask, N, rows, cols);

  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        update_r(d_r, d_tmp, d_imgb, N, sigma, item);
      });
}

void grad(sycl::queue &q, sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
          const size_t rows, const size_t cols, const size_t ndyn, float mu,
          char mode)
  {
    switch (mode)
    {
      case 'x':
      q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_xx(d_out, d_in, N, rows, cols, ndyn, item);
        });
      q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_xx_bound(d_out, N, rows, cols, ndyn, item);
        });
        break;
      case 'y':
      q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_yy(d_out, d_in, N, rows, cols, ndyn, item);
        });
      q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_yy_bound(d_out, N, rows, cols, ndyn, item);
        });
        break;
      case 't':
      q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_tt(d_out, d_in, N, rows, cols, ndyn, mu, item);
        });
      q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_tt_bound(d_out, N, rows, cols, ndyn, item);
        });
        break;
      default:
        printf("Please select x, y or z!\n");
        break;
    }
  }

void proj_p(sycl::float2 *d_p, sycl::float2 *d_tmp, sycl::float2 *d_wbar,
            const size_t N, float sigma, float alpha,
            const sycl::nd_item<3> &item)
{
  float absp, denom;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N; id += item.get_local_range(2) * item.get_group_range(2))
  {
    d_p[id] = d_p[id] - sigma*(d_tmp[id] + d_wbar[id]);
    d_p[id+N] = d_p[id+N] - sigma*(d_tmp[id+N] + d_wbar[id+N]);
    d_p[id+2*N] = d_p[id+2*N] - sigma*(d_tmp[id+2*N] + d_wbar[id+2*N]);

    absp = sycl::sqrt(abs(d_p[id])*abs(d_p[id]) + abs(d_p[id+N])*abs(d_p[id+N])
           + abs(d_p[id+2*N])*abs(d_p[id+2*N]));
    denom = sycl::fmax(1.f, absp/alpha);

    d_p[id] = d_p[id]/denom;
    d_p[id+N] = d_p[id+N]/denom;
    d_p[id+2*N] = d_p[id+2*N]/denom;
  }
}

void grad_xx_adj(sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
                 const size_t rows, const size_t cols, const size_t ndyn,
                 const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N - 2 * rows * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id%(rows*(cols-2)) + id/(rows*(cols-2))*(rows*cols) + rows;

      d_out[idx] = d_in[idx] - d_in[idx-rows];
    }
}

void grad_xx_adj_zero(sycl::float2 *d_out, sycl::float2 *d_in,
                      const size_t rows, const size_t cols, const size_t ndyn,
                      const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < rows * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id%rows + id/rows*(rows*cols);

      d_out[idx] = d_in[idx];
    }
}

void grad_xx_adj_bound(sycl::float2 *d_out, sycl::float2 *d_in,
                       const size_t rows, const size_t cols, const size_t ndyn,
                       const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < rows * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id%rows + id/rows*(rows*cols) + rows*(cols-1);

      d_out[idx] = -1.f*(d_in[idx-rows]);
    }
}

void grad_yy_adj(sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
                 const size_t rows, const size_t cols, const size_t ndyn,
                 const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N - 2 * cols * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id%(rows-2) + id/(rows-2)*rows + 1;

      d_out[idx] = d_in[idx] - d_in[idx-1];
    }
}

void grad_yy_adj_zero(sycl::float2 *d_out, sycl::float2 *d_in,
                      const size_t rows, const size_t cols, const size_t ndyn,
                      const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < cols * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id*rows;

      d_out[idx] = d_in[idx];
    }
}

void grad_yy_adj_bound(sycl::float2 *d_out, sycl::float2 *d_in,
                       const size_t rows, const size_t cols, const size_t ndyn,
                       const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < cols * ndyn;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id*rows + (rows-1);

      d_out[idx] = -1.f*d_in[idx-1];
    }
}

void grad_tt_adj(sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
                 const size_t rows, const size_t cols, const size_t ndyn,
                 float mu, const sycl::nd_item<3> &item)
{
  int stride = rows*cols;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N - 2 * rows * cols;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_out[id+stride] = (d_in[id+stride] - d_in[id])/mu;
    }
}

void grad_tt_adj_zero(sycl::float2 *d_out, sycl::float2 *d_in,
                      const size_t rows, const size_t cols, const size_t ndyn,
                      const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < rows * cols;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id;
      d_out[idx] = d_in[idx];
    }
}

void grad_tt_adj_bound(sycl::float2 *d_out, sycl::float2 *d_in,
                       const size_t rows, const size_t cols, const size_t ndyn,
                       const sycl::nd_item<3> &item)
{
  int idx;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < rows * cols;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      idx = id + rows*cols*(ndyn-1);
      d_out[idx] = -1.f*d_in[idx-rows*cols];
    }
}

void grad_adj(sycl::queue &q, sycl::float2 *d_out, sycl::float2 *d_in, const size_t N,
              const size_t rows, const size_t cols, const size_t ndyn, float mu,
              char mode)
{
  switch (mode)
  {
    case 'x':
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_xx_adj(d_out, d_in, N, rows, cols, ndyn, item);
        });
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_xx_adj_zero(d_out, d_in, rows, cols, ndyn, item);
        });
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_xx_adj_bound(d_out, d_in, rows, cols, ndyn, item);
        });
      break;
    case 'y':
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_yy_adj(d_out, d_in, N, rows, cols, ndyn, item);
        });
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_yy_adj_zero(d_out, d_in, rows, cols, ndyn, item);
        });
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_yy_adj_bound(d_out, d_in, rows, cols, ndyn, item);
        });
      break;
    case 't':
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_tt_adj(d_out, d_in, N, rows, cols, ndyn, mu, item);
        });
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_tt_adj_zero(d_out, d_in, rows, cols, ndyn, item);
        });
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          grad_tt_adj_bound(d_out, d_in, rows, cols, ndyn, item);
        });
      break;
    default:
      printf("Please select x, y or z!\n");
      break;
  }
}

void proj_q(sycl::float2 *d_q, sycl::float2 *d_tmp, const size_t N, float sigma,
            float alpha, const sycl::nd_item<3> &item)
{
  float absq, denom;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N; id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_tmp[id] = d_tmp[id+3*N];
      d_tmp[id+N] = d_tmp[id+4*N];
      d_tmp[id+2*N] = d_tmp[id+5*N];
      d_tmp[id+3*N] = (d_tmp[id+6*N] + d_tmp[id+7*N])/2.f;
      d_tmp[id+4*N] = (d_tmp[id+8*N] + d_tmp[id+9*N])/2.f;
      d_tmp[id+5*N] = (d_tmp[id+10*N] + d_tmp[id+11*N])/2.f;

      d_q[id] = d_q[id] - sigma*d_tmp[id];
      d_q[id+N] = d_q[id+N] - sigma*d_tmp[id+N];
      d_q[id+2*N] = d_q[id+2*N] - sigma*d_tmp[id+2*N];
      d_q[id+3*N] = d_q[id+3*N] - sigma*d_tmp[id+3*N];
      d_q[id+4*N] = d_q[id+4*N] - sigma*d_tmp[id+4*N];
      d_q[id+5*N] = d_q[id+5*N] - sigma*d_tmp[id+5*N];

      absq = sycl::sqrt(abs(d_q[id])*abs(d_q[id])
                 + abs(d_q[id+N])*abs(d_q[id+N])
                 + abs(d_q[id+2*N])*abs(d_q[id+2*N])
                 + 2*abs(d_q[id+3*N])*abs(d_q[id+3*N])
                 + 2*abs(d_q[id+4*N])*abs(d_q[id+4*N])
                 + 2*abs(d_q[id+5*N])*abs(d_q[id+5*N]));
      denom = sycl::fmax(1.f, absq/alpha);

      d_q[id] = d_q[id]/denom;
      d_q[id+N] = d_q[id+N]/denom;
      d_q[id+2*N] = d_q[id+2*N]/denom;
      d_q[id+3*N] = d_q[id+3*N]/denom;
      d_q[id+4*N] = d_q[id+4*N]/denom;
      d_q[id+5*N] = d_q[id+5*N]/denom;
    }
}

void update_s(sycl::float2 *d_imgs, sycl::float2 *d_tmp, sycl::float2 *d_imgz,
              const size_t N, float tau, const sycl::nd_item<3> &item)
{
  sycl::float2 divp;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N; id += item.get_local_range(2) * item.get_group_range(2))
    {
      divp = d_tmp[id] + d_tmp[id+N] + d_tmp[id+2*N];
      d_imgs[id] = d_imgs[id] - tau*(d_imgz[id] + divp);
    }
}

void update_w(sycl::float2 *d_w, sycl::float2 *d_tmp, sycl::float2 *d_p,
              const size_t N, float tau, const sycl::nd_item<3> &item)
{
  sycl::float2 divq;
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < N; id += item.get_local_range(2) * item.get_group_range(2))
    {
      divq = d_tmp[id+3*N] + d_tmp[id+4*N] + d_tmp[id+5*N];
      d_w[id] = d_w[id] - tau*(divq - d_p[id]);

      divq = d_tmp[id+6*N] + d_tmp[id+7*N] + d_tmp[id+8*N];
      d_w[id+N] = d_w[id+N] - tau*(divq - d_p[id+N]);

      divq = d_tmp[id+9*N] + d_tmp[id+10*N] + d_tmp[id+11*N];
      d_w[id+2*N] = d_w[id+2*N] - tau*(divq - d_p[id+2*N]);
    }
}

void shrink(sycl::float2 *d_array2, float *d_array, const float beta,
            const int array_size, const sycl::nd_item<3> &item)
{
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < array_size;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_array[id] = sycl::fmax(d_array[id] - beta, 0.f);
      d_array2[id] = make_float2(d_array[id]);
    }
}

void arrayabs(float *d_array, sycl::float2 *d_array2, const size_t array_size,
              const sycl::nd_item<3> &item)
{
  for (int id = item.get_group(2) * item.get_local_range(2) +
                item.get_local_id(2);
       id < array_size;
       id += item.get_local_range(2) * item.get_group_range(2))
    {
      d_array[id] = abs(d_array2[id]);
    }
}

float compute_ser(sycl::queue &q, sycl::float2 *d_array, sycl::float2 *d_img,
                  const size_t array_size) {
  float *d_diff, *d_tmp, ser, diff_norm, img_norm;
  float *d_img_norm, *d_diff_norm;

  d_diff = sycl::malloc_device<float>(array_size, q);
  d_tmp = sycl::malloc_device<float>(array_size, q);
  d_img_norm = sycl::malloc_device<float>(1, q);
  d_diff_norm = sycl::malloc_device<float>(1, q);

  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        arrayabs(d_diff, d_array, array_size, item);
      });
  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        arrayabs(d_tmp, d_img, array_size, item);
      });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                           sycl::range<3>(1, 1, gridsize),
                                       sycl::range<3>(1, 1, gridsize)),
                     [=](sycl::nd_item<3> item) {
                       arrayadd(d_diff, d_diff, d_tmp, array_size, 1.f, -1.f,
                                item);
                     });
  });

  oneapi::mkl::blas::column_major::nrm2(q, array_size, d_tmp, 1, d_img_norm);
  q.memcpy(&img_norm, d_img_norm, sizeof(float));
  oneapi::mkl::blas::column_major::nrm2(q, array_size, d_diff, 1, d_diff_norm);
  q.memcpy(&diff_norm, d_diff_norm, sizeof(float));
  q.wait();

  ser = -20.f*log10f(diff_norm/img_norm);

  sycl::free(d_diff, q);
  sycl::free(d_tmp, q);
  sycl::free(d_diff_norm, q);
  sycl::free(d_img_norm, q);

  return ser;
}

void tgv_cs(sycl::queue &q,
            sycl::float2 *d_imgl, sycl::float2 *d_imgs, sycl::float2 *h_img,
            sycl::float2 *h_mask, const size_t N, const size_t rows,
            const size_t cols, const size_t ndyn, float alpha, float beta,
            float mu, float tau, float sigma, float reduction, int iter) {

  // Read image and mask
  sycl::float2 *d_img, *d_imgz, *d_imgb, *d_mask;

  d_img = sycl::malloc_device<sycl::float2>(N, q);
  q.memcpy(d_img, h_img, N * sizeof(sycl::float2));

  d_mask = sycl::malloc_device<sycl::float2>(N, q);
  q.memcpy(d_mask, h_mask, N * sizeof(sycl::float2));

  // Scale the data to [0,1]
  float img_max = compute_maxmag(q, d_img, N);
  q.submit([&](sycl::handler &cgh) {
    auto img_max_ct2 = 1.f / img_max;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                           sycl::range<3>(1, 1, gridsize),
                                       sycl::range<3>(1, 1, gridsize)),
                     [=](sycl::nd_item<3> item) {
                       scaledata(d_img, N, img_max_ct2, item);
                     });
  });

  // Sample image using mask and generate zerofilled image
  d_imgz = sycl::malloc_device<sycl::float2>(N, q);
  d_imgb = sycl::malloc_device<sycl::float2>(N, q);

  fft2_init(q, rows, cols, ndyn);

  forward(q, d_imgb, d_img, d_mask, N, rows, cols);
  backward(q, d_imgz, d_imgb, d_mask, N, rows, cols);

  // Allocate arrays
  sycl::float2 *d_w;                      // Primal variables
  sycl::float2 *d_p, *d_q, *d_r;          // Dual variables
  sycl::float2 *d_lbar, *d_sbar, *d_wbar; // Intermediate variables
  sycl::float2 *d_lold, *d_sold, *d_wold; // Old variables
  sycl::float2 *d_tmp; // Temporal variable (grad, grad_adj ect)

  d_w = sycl::malloc_device<sycl::float2>(3 * N, q);
  d_p = sycl::malloc_device<sycl::float2>(3 * N, q);
  d_q = sycl::malloc_device<sycl::float2>(6 * N, q);
  d_r = sycl::malloc_device<sycl::float2>(N, q);
  d_lbar = sycl::malloc_device<sycl::float2>(N, q);
  d_sbar = sycl::malloc_device<sycl::float2>(N, q);
  d_wbar = sycl::malloc_device<sycl::float2>(3 * N, q);
  d_lold = sycl::malloc_device<sycl::float2>(N, q);
  d_sold = sycl::malloc_device<sycl::float2>(N, q);
  d_wold = sycl::malloc_device<sycl::float2>(3 * N, q);
  d_tmp = sycl::malloc_device<sycl::float2>(12 * N, q);

  // Initialization for variables
  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        arrayreal(d_imgl, d_imgz, N, item);
      });
  q.memset(d_imgs, 0.f, N * sizeof(sycl::float2));
  q.memset(d_w, 0.f, 3 * N * sizeof(sycl::float2));
  q.memcpy(d_sbar, d_imgs, N * sizeof(sycl::float2));
  q.memcpy(d_lbar, d_imgl, N * sizeof(sycl::float2));
  q.memcpy(d_wbar, d_w, 3 * N * sizeof(sycl::float2));
  q.memset(d_p, 0.f, 3 * N * sizeof(sycl::float2));
  q.memset(d_q, 0.f, 6 * N * sizeof(sycl::float2));
  q.memset(d_r, 0.f, N * sizeof(sycl::float2));
  q.memset(d_tmp, 0.f, 12 * N * sizeof(sycl::float2));

  // Initialization for alpha
  float alpha0 = sqrtf(2)*alpha;
  float alpha1 = alpha;

  float alpha00 = alpha0;
  float alpha10 = alpha1;
  float alpha01 = alpha0*reduction;
  float alpha11 = alpha1*reduction;

  // Initialization for SVD
  int lwork, lda, ldu, lds, ldvt;

  lda = rows*cols;
  ldu = rows*cols;
  lds = fminf(ndyn, rows*cols);
  ldvt = ndyn;

  oneapi::mkl::jobsvd jobu = oneapi::mkl::jobsvd::somevec;
  oneapi::mkl::jobsvd jobvt = oneapi::mkl::jobsvd::somevec;
  lwork = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(
              q, jobu, jobvt, rows * cols, ndyn, lda, ldu, ldvt);

  sycl::float2 *d_lu, *d_ls2, *d_lvt, *d_lsvt, *d_work, a, b;
  float *d_ls, *d_rwork;

  d_lu = sycl::malloc_device<sycl::float2>(ldu * lds, q);
  d_ls = sycl::malloc_device<float>(lds, q);
  d_ls2 = sycl::malloc_device<sycl::float2>(lds, q);
  d_lvt = sycl::malloc_device<sycl::float2>(lds * ldvt, q);
  d_lsvt = sycl::malloc_device<sycl::float2>(lds * ldvt, q);
  d_work = sycl::malloc_device<sycl::float2>(lwork, q);
  d_rwork = sycl::malloc_device<float>((lds - 1), q);

  a = make_float2(1.f);
  b = make_float2(0.f);

  // Run the main loop
  printf("Running recon ...\n");
  for (int i = 1; i <= iter; i++)
  {
    // update alpha
    alpha0 = compute_alpha(alpha00, alpha01, iter, i);
    alpha1 = compute_alpha(alpha10, alpha11, iter, i);

    
    q.memcpy(d_lold, d_imgl, N * sizeof(sycl::float2));
    q.memcpy(d_sold, d_imgs, N * sizeof(sycl::float2));
    q.memcpy(d_wold, d_w, 3 * N * sizeof(sycl::float2));

    // update r
    update_r(q, d_r, d_lbar, d_sbar, d_imgb, d_tmp, d_mask,
      sigma, N, rows, cols);

    // update p
    grad(q, d_tmp, d_sbar, N, rows, cols, ndyn, mu, 'x');
    grad(q, d_tmp+N, d_sbar, N, rows, cols, ndyn, mu, 'y');
    grad(q, d_tmp+2*N, d_sbar, N, rows, cols, ndyn, mu, 't');

    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          proj_p(d_p, d_tmp, d_wbar, N, sigma, alpha1, item);
        });

    // update q
    grad_adj(q, d_tmp+3*N, d_wbar, N, rows, cols, ndyn, mu, 'x');
    grad_adj(q, d_tmp+4*N, d_wbar+N, N, rows, cols, ndyn, mu, 'y');
    grad_adj(q, d_tmp+5*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 't');
    grad_adj(q, d_tmp+6*N, d_wbar, N, rows, cols, ndyn, mu, 'y');
    grad_adj(q, d_tmp+7*N, d_wbar+N, N, rows, cols, ndyn, mu, 'x');
    grad_adj(q, d_tmp+8*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 'x');
    grad_adj(q, d_tmp+9*N, d_wbar, N, rows, cols, ndyn, mu, 't');
    grad_adj(q, d_tmp+10*N, d_wbar+N, N, rows, cols, ndyn, mu, 't');
    grad_adj(q, d_tmp+11*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 'y');

    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          proj_q(d_q, d_tmp, N, sigma, alpha0, item);
        });

    // update s
    backward(q, d_imgz, d_r, d_mask, N, rows, cols);

    grad_adj(q, d_tmp, d_p, N, rows, cols, ndyn, mu, 'x');
    grad_adj(q, d_tmp+N, d_p+N, N, rows, cols, ndyn, mu, 'y');
    grad_adj(q, d_tmp+2*N, d_p+2*N, N, rows, cols, ndyn, mu, 't');

    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          update_s(d_imgs, d_tmp, d_imgz, N, tau, item);
        });

    // update w
    grad(q, d_tmp+3*N, d_q, N, rows, cols, ndyn, mu, 'x');
    grad(q, d_tmp+4*N, d_q+3*N, N, rows, cols, ndyn, mu, 'y');
    grad(q, d_tmp+5*N, d_q+4*N, N, rows, cols, ndyn, mu, 't');
    grad(q, d_tmp+6*N, d_q+3*N, N, rows, cols, ndyn, mu, 'x');
    grad(q, d_tmp+7*N, d_q+N, N, rows, cols, ndyn, mu, 'y');
    grad(q, d_tmp+8*N, d_q+5*N, N, rows, cols, ndyn, mu, 't');
    grad(q, d_tmp+9*N, d_q+4*N, N, rows, cols, ndyn, mu, 'x');
    grad(q, d_tmp+10*N, d_q+5*N, N, rows, cols, ndyn, mu, 'y');
    grad(q, d_tmp+11*N, d_q+2*N, N, rows, cols, ndyn, mu, 't');

    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          update_w(d_w, d_tmp, d_p, N, tau, item);
        });

    // update l
    q.submit([&](sycl::handler &cgh) {
      auto tau_ct5 = -1.f * tau;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                             sycl::range<3>(1, 1, gridsize),
                                         sycl::range<3>(1, 1, gridsize)),
                       [=](sycl::nd_item<3> item) {
                         arrayadd(d_imgl, d_imgl, d_imgz, N, 1.f, tau_ct5,
                                  item);
                       });
    });

    oneapi::mkl::lapack::gesvd(
        q, jobu, jobvt,
        rows * cols, ndyn, (std::complex<float> *)d_imgl, lda, (float *)d_ls,
        (std::complex<float> *)d_lu, ldu, (std::complex<float> *)d_lvt, ldvt,
        (std::complex<float> *)d_work, lwork);

    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                              sycl::range<3>(1, 1, gridsize),
                          sycl::range<3>(1, 1, gridsize)),
        [=](sycl::nd_item<3> item) {
          shrink(d_ls2, d_ls, beta, lds, item);
        });

    oneapi::mkl::blas::column_major::dgmm(
        q, oneapi::mkl::side::left, lds, ldvt,
        (std::complex<float> *)d_lvt, ldvt, (std::complex<float> *)d_ls2, 1,
        (std::complex<float> *)d_lsvt, ldvt);

    oneapi::mkl::blas::column_major::gemm(
        q, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, ldu, ldvt, lds,
        std::complex<float>(a.x(), a.y()), (std::complex<float> *)d_lu, ldu,
        (std::complex<float> *)d_lsvt, ldvt, std::complex<float>(b.x(), b.y()),
        (std::complex<float> *)d_imgl, ldu);

    // update intermediate variables
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                             sycl::range<3>(1, 1, gridsize),
                                         sycl::range<3>(1, 1, gridsize)),
                       [=](sycl::nd_item<3> item) {
                         arrayadd(d_sbar, d_imgs, d_sold, N, 2.f, -1.f,
                                  item);
                       });
    });
    q.submit([&](sycl::handler &cgh) {
      auto N_ct3 = 3 * N;
      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                             sycl::range<3>(1, 1, gridsize),
                                         sycl::range<3>(1, 1, gridsize)),
                       [=](sycl::nd_item<3> item) {
                         arrayadd(d_wbar, d_w, d_wold, N_ct3, 2.f, -1.f,
                                  item);
                       });
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                                             sycl::range<3>(1, 1, gridsize),
                                         sycl::range<3>(1, 1, gridsize)),
                       [=](sycl::nd_item<3> item) {
                         arrayadd(d_lbar, d_imgl, d_lold, N, 2.f, -1.f,
                                  item);
                       });
    });
  }

  // Compute SER
  sycl::float2 *d_imgr;
  d_imgr = sycl::malloc_device<sycl::float2>(N, q);

  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocksize) *
                            sycl::range<3>(1, 1, gridsize),
                        sycl::range<3>(1, 1, gridsize)),
      [=](sycl::nd_item<3> item) {
        arrayadd(d_imgr, d_imgl, d_imgs, N, 1.f, 1.f, item);
      });
  backward(q, d_imgz, d_imgb, d_mask, N, rows, cols);

  float ser_imgz, ser_imgr;
  ser_imgz = compute_ser(q, d_imgz, d_img, N);
  ser_imgr = compute_ser(q, d_imgr, d_img, N);

  printf("The SER of zerofill: %.2f dB\n", ser_imgz);
  printf("The SER of recon:    %.2f dB\n", ser_imgr);

  // Free arrays and destroy handles
  sycl::free(d_img, q);
  sycl::free(d_imgb, q);
  sycl::free(d_imgz, q);
  sycl::free(d_imgr, q);
  sycl::free(d_mask, q);

  sycl::free(d_w, q);
  sycl::free(d_p, q);
  sycl::free(d_q, q);
  sycl::free(d_r, q);
  sycl::free(d_lbar, q);
  sycl::free(d_sbar, q);
  sycl::free(d_wbar, q);
  sycl::free(d_lold, q);
  sycl::free(d_sold, q);
  sycl::free(d_wold, q);
  sycl::free(d_tmp, q);

  sycl::free(d_lu, q);
  sycl::free(d_ls, q);
  sycl::free(d_ls2, q);
  sycl::free(d_lvt, q);
  sycl::free(d_lsvt, q);
  sycl::free(d_work, q);
  sycl::free(d_rwork, q);
}
