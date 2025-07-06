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

#include <chrono>
#include "tgv.h"

extern int blocksize;
extern int gridsize;

int
main (int argc, char *argv[])
{
    // Setup default values
    const char *in_img, *in_mask, *out_img;
    in_img = "../../tgvnn-cuda/data/pincat.ra";
    in_mask = "../../tgvnn-cuda/data/pincat_mask.ra";
    out_img = "../result/pincat_recon.ra";

    float alpha = 0.004;
    float beta = 0.5;
    float sigma = 0.25;
    float tau = 0.25;
    float mu = 1.f;
    float reduction = 0.5; // No need to change this
    int iter = 500;

    // Read command line
    struct option long_options[] =
    {
        {"iter", 1, NULL, 'i'},
        {"alpha", 1, NULL, 'a'},
        {"beta", 1, NULL, 'b'},
        {"sigma", 1, NULL, 's'},
        {"tau", 1, NULL, 't'},
        {"mu", 1, NULL, 'm'},
        {"output", 1, NULL, 'o'},
        {"gridsize", 1, NULL, 'G'},
        {"blocksize", 1, NULL, 'B'},
        {"help", 0, 0, 'h'}
    };

    extern int optind;
    opterr = 0;
    int option_index = 0;
    int c;
    while ((c =
      getopt_long(argc, argv, "i:a:b:s:t:m:o:G:B:h",
        long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'i':
                iter = atoi(optarg);
                break;
            case 'a':
                alpha = atof(optarg);
                break;
            case 'b':
                beta = atof(optarg);
                break;
            case 's':
                sigma = atof(optarg);
                break;
            case 't':
                tau = atof(optarg);
                break;
            case 'm':
                mu = atof(optarg);
            case 'o':
                out_img = optarg;
                break;
            case 'G':
                gridsize = atoi(optarg);
                break;
            case 'B':
                blocksize = atoi(optarg);
                break;
            case 'h':
            default:
                print_usage();
                return 1;
        }
    }

    int ra_count = 0;
    const char *tmp;
    while (optind <= argc-1)
    {
      tmp = argv[optind];

      if (strstr(tmp, ".ra") != NULL)
      {
        if (ra_count == 0)
        {
          in_img = tmp;
          ra_count ++;
          optind ++;
        }
        else if (ra_count == 1)
        {
          in_mask = tmp;
          ra_count ++;
          optind ++;
        }
        else
        {
          break;
        }
      }
      else
      {
        optind ++;
      }
    }

    // Read input data and mask
    printf("Reading input ...\n");
    printf("Input image: %s\n", in_img);
    printf("Input mask:  %s\n", in_mask);
    printf("gridsize:  %d\n", gridsize);
    printf("blocksize: %d\n", blocksize);
    printf("alpha: %.4f\n", alpha);
    printf("beta:  %.4f\n", beta);
    printf("sigma: %.4f\n", sigma);
    printf("tau:   %.4f\n", tau);
    printf("iter:  %d\n", iter);

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    ra_t ra_img, ra_mask;

    ra_read(&ra_img, in_img);
    ra_read(&ra_mask, in_mask);

    sycl::float2 *h_img = (sycl::float2 *)ra_img.data;
    sycl::float2 *h_mask = (sycl::float2 *)ra_mask.data;

    size_t rows = ra_img.dims[0];
    size_t cols = ra_img.dims[1];
    size_t ndyn = ra_img.dims[2];
    size_t N = ndyn*rows*cols;

    printf("rows = %lu, cols = %lu, ndyn = %lu, N = %lu\n", rows, cols, ndyn, N);
    printf("\n");

    // Run TGV+NN
    sycl::float2 *d_imgl, *d_imgs;
    d_imgl = sycl::malloc_device<sycl::float2>(N, q);
    d_imgs = sycl::malloc_device<sycl::float2>(N, q);

    auto start = std::chrono::steady_clock::now();

    tgv_cs(q, d_imgl, d_imgs, h_img, h_mask, N, rows, cols, ndyn,
           alpha, beta, mu, tau, sigma, reduction, iter);
    q.wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total execution time of tgv: %f (s)\n", time * 1e-9f);

    // Save results
    printf("Saving output ...\n");
    printf("Output: %s\n", out_img);

    sycl::float2 *h_out;
    h_out = (sycl::float2 *)safe_malloc(2 * N * sizeof(sycl::float2));
    q.memcpy(h_out, d_imgl, N * sizeof(sycl::float2));
    q.memcpy(h_out + N, d_imgs, N * sizeof(sycl::float2));
    q.wait();
    save_rafile(h_out, out_img, rows, cols, ndyn, 2);

    // Free memory
    ra_free(&ra_img);
    ra_free(&ra_mask);
    safe_free(h_out);
    sycl::free(d_imgl, q);
    sycl::free(d_imgs, q);

    return 0;
}
