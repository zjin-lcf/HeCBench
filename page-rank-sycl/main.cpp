/*
 ** The MIT License (MIT)
 **
 ** Copyright (c) 2014, Erick Lavoie, Faiz Khan, Sujay Kathrotia, Vincent
 ** Foley-Bourgon, Laurie Hendren
 **
 ** Permission is hereby granted, free of charge, to any person obtaining a copy
 **of this software and associated documentation files (the "Software"), to deal
 ** in the Software without restriction, including without limitation the rights
 ** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 ** copies of the Software, and to permit persons to whom the Software is
 ** furnished to do so, subject to the following conditions:
 **
 ** The above copyright notice and this permission notice shall be included in all
 ** copies or substantial portions of the Software.
 **
 ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 ** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 ** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 ** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 ** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 ** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 ** SOFTWARE.
 **
 **/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <chrono>
#include <sycl/sycl.hpp>

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

#define D_FACTOR (0.85f)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// default values
const int max_iter = 1000;
const float threshold= 1e-16f;

// generates an array of random pages and their links
int *random_pages(int n, unsigned int *noutlinks, int divisor){
  int i, j, k;
  int *pages = (int*) malloc(sizeof(int)*n*n); // matrix 1 means link from j->i

  if (divisor <= 0) {
    fprintf(stderr, "ERROR: Invalid divisor '%d' for random initialization, divisor should be greater or equal to 1\n", divisor);
    exit(1);
  }

  for(i=0; i<n; ++i){
    noutlinks[i] = 0;
    for(j=0; j<n; ++j){
      if(i!=j && (abs(rand())%divisor == 0)){
        pages[i*n+j] = 1;
        noutlinks[i] += 1;
      }
    }

    // the case with no outlinks is avoided
    if(noutlinks[i] == 0){
      do { k = abs(rand()) % n; } while ( k == i);
      pages[i*n + k] = 1;
      noutlinks[i] = 1;
    }
  }
  return pages;
}

void init_array(float *a, int n, float val){
  int i;
  for(i=0; i<n; ++i){
    a[i] = val;
  }
}

void usage(char *argv[]){
  fprintf(stderr, "Usage: %s [-n number of pages] [-i max iterations]"
                  " [-t threshold] [-q divsor for zero density]\n", argv[0]);
}

static struct option size_opts[] =
{
  /* name, has_tag, flag, val*/
  {"number of pages", 1, NULL, 'n'},
  {"max number of iterations", 1, NULL, 'i'},
  {"minimum threshold", 1, NULL, 't'},
  {"divisor for zero density", 1, NULL, 'q'},
  { 0, 0, 0}
};

float maximum_dif(float *difs, int n){
  int i;
  float max = 0.0f;
  for(i=0; i<n; ++i){
    max = difs[i] > max ? difs[i] : max;
  }
  return max;
}

int main(int argc, char *argv[]) {
  int *pages;
  float *maps;
  float *page_ranks;
  unsigned int *noutlinks;
  int t;
  float max_diff;

  int i = 0;
  int j;
  int n = 1000;
  int iter = max_iter;
  float thresh = threshold;
  int divisor = 2;
  int nb_links = 0;

  int opt, opt_index = 0;
  while((opt = getopt_long(argc, argv, "::n:i:t:q:", size_opts, &opt_index)) != -1){
    switch(opt){
      case 'n':
        n = atoi(optarg);
        break;
      case 'i':
        iter = atoi(optarg);
        break;
      case 't':
        thresh = atof(optarg);
        break;
      case 'q':
        divisor = atoi(optarg);
        break;
      default:
        usage(argv);
        exit(EXIT_FAILURE);
    }
  }
  page_ranks = (float*)malloc(sizeof(float)*n);
  maps = (float*)malloc(sizeof(float)*n*n);
  noutlinks = (unsigned int*)malloc(sizeof(unsigned int)*n);

  max_diff=99.0f;

  for (i=0; i<n; ++i) {
    noutlinks[i] = 0;
  }
  pages = random_pages(n,noutlinks,divisor);
  init_array(page_ranks, n, 1.0f / (float) n);

  nb_links = 0;
  for (i=0; i<n; ++i) {
    for (j=0; j<n; ++j) {
      nb_links += pages[i*n+j];
    }
  }

  float *diffs;
  diffs  = (float*) malloc(sizeof(float)*n);
  for(i = 0; i < n; ++i){
    diffs[i] = 0.0f;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_pages = sycl::malloc_device<int>(n*n, q);
  q.memcpy(d_pages, pages, sizeof(int) * n * n);

  float *d_page_ranks = sycl::malloc_device<float>(n, q);
  q.memcpy(d_page_ranks, page_ranks, sizeof(float) * n);

  float *d_maps = sycl::malloc_device<float>(n*n, q);

  unsigned int *d_noutlinks = sycl::malloc_device<unsigned int>(n, q);
  q.memcpy(d_noutlinks, noutlinks, sizeof(unsigned int) * n);

  float *d_diffs = sycl::malloc_device<float>(n, q);

  size_t block_size  = n < BLOCK_SIZE ? n : BLOCK_SIZE;
  size_t global_work_size = (n+block_size-1) / block_size * block_size;

  sycl::range<1> gws (global_work_size);
  sycl::range<1> lws (block_size);

  q.wait();
  double ktime = 0.0;

  for (t=1; t<=iter && max_diff>=thresh; ++t) {
    auto start = std::chrono::high_resolution_clock::now();

    //map <<< dim3(num_blocks), dim3(block_size) >>> ( d_pages, d_page_ranks, d_maps, d_noutlinks, n);
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class map>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          float outbound_rank = ldg(&d_page_ranks[i])/(float)ldg(&d_noutlinks[i]);
          for(int j=0; j<n; ++j)
            d_maps[i*n+j] = ldg(&d_pages[i*n+j])*outbound_rank;
        }
      });
    });

    //reduce<<< dim3(num_blocks), dim3(block_size) >>>(d_page_ranks, d_maps, n, d_diffs);
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class reduce>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int j = item.get_global_id(0);
        float new_rank;
        float old_rank;
        if (j < n) {
          old_rank = d_page_ranks[j];
          new_rank = 0.0f;
          for(int i=0; i< n; ++i) new_rank += d_maps[i*n + j];
          new_rank = ((1.f-D_FACTOR)/n)+(D_FACTOR*new_rank);
          d_diffs[j] = sycl::max(sycl::fabs(new_rank - old_rank), d_diffs[j]);
          d_page_ranks[j] = new_rank;
        }
      });
    });

    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    q.memcpy(diffs, d_diffs, sizeof(float)*n).wait();
    q.memset(d_diffs, 0, sizeof(float)*n);

    max_diff = maximum_dif(diffs, n);
  }

  sycl::free(d_pages, q);
  sycl::free(d_maps, q);
  sycl::free(d_page_ranks, q);
  sycl::free(d_noutlinks, q);
  sycl::free(d_diffs, q);

  fprintf(stderr, "Max difference %f is reached at iteration %d\n", max_diff, t);
  printf("\"Options\": \"-n %d -i %d -t %f\". Total kernel execution time: %lf (s)\n",
         n, iter, thresh, ktime);

  free(pages);
  free(maps);
  free(page_ranks);
  free(noutlinks);
  free(diffs);
  return 0;
}
