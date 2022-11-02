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
#include "common.h"

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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  const property_list props = property::buffer::use_host_ptr();

  buffer<int, 1> d_pages (pages, n*n, props);
  buffer<float, 1> d_page_ranks (page_ranks, n, props);
  buffer<float, 1> d_maps (maps, n*n, props);
  buffer<unsigned int, 1> d_noutlinks (noutlinks, n, props);
  buffer<float, 1> d_diffs (diffs, n, props);

  d_page_ranks.set_final_data(nullptr);
  d_maps.set_final_data(nullptr);
  d_diffs.set_final_data(nullptr);

  size_t block_size  = n < BLOCK_SIZE ? n : BLOCK_SIZE;
  size_t global_work_size = (n+block_size-1) / block_size * block_size;

  range<1> gws (global_work_size);
  range<1> lws (block_size);

  double ktime = 0.0;

  for (t=1; t<=iter && max_diff>=thresh; ++t) {
    auto start = std::chrono::high_resolution_clock::now();

    //map <<< dim3(num_blocks), dim3(block_size) >>> ( d_pages, d_page_ranks, d_maps, d_noutlinks, n);
    q.submit([&](handler& cgh) { 
      auto pages = d_pages.get_access<sycl_read>(cgh);
      auto page_ranks = d_page_ranks.get_access<sycl_read>(cgh);
      auto maps = d_maps.get_access<sycl_discard_write>(cgh);
      auto noutlinks = d_noutlinks.get_access<sycl_read>(cgh);
      cgh.parallel_for<class map>( 
        nd_range<1>(range<1>(global_work_size), range<1>(block_size)), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          float outbound_rank = page_ranks[i]/(float)noutlinks[i];
          for(int j=0; j<n; ++j) maps[i*n+j] = pages[i*n+j]*outbound_rank;
        }
      });
    });

    //reduce<<< dim3(num_blocks), dim3(block_size) >>>(d_page_ranks, d_maps, n, d_diffs);
    q.submit([&](handler& cgh) { 
      auto page_ranks = d_page_ranks.get_access<sycl_discard_read_write>(cgh);
      auto maps = d_maps.get_access<sycl_discard_read_write>(cgh);
      auto dif = d_diffs.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class reduce>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        int j = item.get_global_id(0);
        float new_rank;
        float old_rank;
        if (j < n) {
          old_rank = page_ranks[j];
          new_rank = 0.0f;
          for(int i=0; i< n; ++i) new_rank += maps[i*n + j];
          new_rank = ((1.f-D_FACTOR)/n)+(D_FACTOR*new_rank);
          dif[j] = sycl::fabs(new_rank - old_rank) > dif[j] ? 
                   sycl::fabs(new_rank - old_rank) : dif[j];
          page_ranks[j] = new_rank;
        }
      });
    });

    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    q.submit([&](handler& cgh) { 
      auto diffs_acc = d_diffs.get_access<sycl_read>(cgh);
      cgh.copy(diffs_acc, diffs);
    }).wait();

    q.submit([&](handler& cgh) { 
      auto diffs_acc = d_diffs.get_access<sycl_discard_write>(cgh);
      cgh.fill(diffs_acc, 0.f);
    });

    max_diff = maximum_dif(diffs, n);
  }

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
