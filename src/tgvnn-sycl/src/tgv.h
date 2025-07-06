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

#ifndef _TGV_H
#define _TGV_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <sycl/sycl.hpp>
#include "fft.h"
#include "float2math.h"
#include "ra.h"

#ifdef __cplusplus
#endif

void * mallocAssert(const size_t n, const char *file, int line);
void freeAssert(void *p, const char *file, int line);
void print_usage();
void save_rafile(sycl::float2 *h_out, const char *out_path, const size_t dim0,
                 const size_t dim1, const size_t dim2, const size_t dim3);

void tgv_cs(sycl::queue &q, sycl::float2 *d_imgl, sycl::float2 *d_imgs, sycl::float2 *h_img,
            sycl::float2 *h_mask, const size_t N, const size_t rows,
            const size_t cols, const size_t ndyn, float alpha, float beta,
            float mu, float tau, float sigma, float reduction, int iter);

#ifdef __cplusplus
#endif

#ifndef MACROS_H_
#define MACROS_H_

#define safe_malloc(n) mallocAssert(n, __FILE__, __LINE__);
#define safe_free(x) freeAssert(x, __FILE__, __LINE__);

#endif

#endif
