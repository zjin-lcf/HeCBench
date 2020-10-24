/* The MIT License
   Copyright (c) 2011 by Attractive Chaos <attractor@live.co.uk>
   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:
   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "read_data.h"

static void check(int a, int b, const char *s)
{
  if (a != b) printf("Error: %s %d %d\n", s, a, b);
}

typedef struct dpct_type_80c3bf {
  int h, e;
} eh_t;

  void
kernel_extend2(
    const unsigned char* query,
    const unsigned char* target,
    const char* mat,
    eh_t* eh,
    char* qp,
    int* qle_acc,
    int* tle_acc,
    int* gtle_acc,
    int* gscore_acc,
    int* max_off_acc,
    int* score_acc,
    const int qlen, 
    const int tlen, 
    const int m, 
    const int o_del, 
    const int e_del, 
    const int o_ins, 
    const int e_ins, 
    int w, 
    const int end_bonus, 
    const int zdrop, 
    const int h0)
{

  int oe_del = o_del + e_del;
  int oe_ins = o_ins + e_ins; 
  int i, j, k;
  int beg, end;
  int max, max_i, max_j, max_ins, max_del, max_ie;
  int gscore;
  int max_off;

  // generate the query profile
  for (k = i = 0; k < m; ++k) {
    const char *p = mat + k * m;
    for (j = 0; j < qlen; ++j)
      qp[i++] = p[query[j]];
  }

  // fill the first row
  eh[0].h = h0; 
  eh[1].h = h0 > oe_ins? h0 - oe_ins : 0;

  for (j = 2; j <= qlen && eh[j-1].h > e_ins; ++j)
    eh[j].h = eh[j-1].h - e_ins;

  // adjust $w if it is too large
  k = m * m;
  for (i = 0, max = 0; i < k; ++i) // get the max score
    max = max > mat[i]? max : mat[i];
  max_ins = (int)((double)(qlen * max + end_bonus - o_ins) / e_ins + 1.);
  max_ins = max_ins > 1? max_ins : 1;
  w = w < max_ins? w : max_ins;
  max_del = (int)((double)(qlen * max + end_bonus - o_del) / e_del + 1.);
  max_del = max_del > 1? max_del : 1;
  w = w < max_del? w : max_del; // TODO: is this necessary?
  // DP loop
  max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1;
  max_off = 0;
  beg = 0, end = qlen;
  for (i = 0; i < tlen; ++i) {
    int t, f = 0, h1, m = 0, mj = -1;
    char *q = qp + target[i] * qlen;

    // apply the band and the constraint (if provided)
    if (beg < i - w) beg = i - w;
    if (end > i + w + 1) end = i + w + 1;
    if (end > qlen) end = qlen;

    // compute the first column
    if (beg == 0) {
      h1 = h0 - (o_del + e_del * (i + 1));
      if (h1 < 0) h1 = 0;
    } 
    else 
      h1 = 0;

    for (j = beg; j < end; ++j) {
      // At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
      // Similar to SSE2-SW, cells are computed in the following order:
      //   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
      //   E(i+1,j) = max{H(i,j)-gapo, E(i,j)} - gape
      //   F(i,j+1) = max{H(i,j)-gapo, F(i,j)} - gape
      eh_t *p = eh+j;
      int h, M = p->h, e = p->e; // get H(i-1,j-1) and E(i-1,j)
      p->h = h1;          // set H(i,j-1) for the next row
      M = M? M + q[j] : 0;// separating H and M to disallow a cigar like "100M3I3D20M"
      h = M > e? M : e;   // e and f are guaranteed to be non-negative, so h>=0 even if M<0
      h = h > f? h : f;
      h1 = h;             // save H(i,j) to h1 for the next column
      mj = m > h? mj : j; // record the position where max score is achieved
      m = m > h? m : h;   // m is stored at eh[mj+1]
      t = M - oe_del;
      t = t > 0? t : 0;
      e -= e_del;
      e = e > t? e : t;   // computed E(i+1,j)
      p->e = e;           // save E(i+1,j) for the next row
      t = M - oe_ins;
      t = t > 0? t : 0;
      f -= e_ins;
      f = f > t? f : t;   // computed F(i,j+1)
    }
    eh[end].h = h1; eh[end].e = 0;
    if (j == qlen) {
      max_ie = gscore > h1? max_ie : i;
      gscore = gscore > h1? gscore : h1;
    }
    if (m == 0) break;
    if (m > max) {
      max = m, max_i = i, max_j = mj;
   max_off = max_off > sycl::abs(mj - i) ? max_off : sycl::abs(mj - i);
    } else if (zdrop > 0) {
      if (i - max_i > mj - max_j) {
        if (max - m - ((i - max_i) - (mj - max_j)) * e_del > zdrop) break;
      } else {
        if (max - m - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) break;
      }
    }
    // update beg and end for the next round
    for (j = beg; j < end && eh[j].h == 0 && eh[j].e == 0; ++j);
    beg = j;
    for (j = end; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j);
    end = j + 2 < qlen? j + 2 : qlen;
    //beg = 0; end = qlen; // uncomment this line for debugging
  }
  *qle_acc = max_j + 1;
  *tle_acc = max_i + 1;
  *gtle_acc = max_ie + 1;
  *gscore_acc = gscore;
  *max_off_acc = max_off;
  *score_acc = max;
}


void extend2(struct extend2_dat *d)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
  eh_t *eh = NULL; /* score array*/
  char *qp = NULL; /* query profile*/
  posix_memalign((void**)&eh, 64, (d->qlen+1) * 8);
  posix_memalign((void**)&qp, 64, d->qlen * d->m);
  memset(eh, 0, (d->qlen+1) * 8);

  int qle, tle, gtle, gscore, max_off, score;

  const int qlen = d->qlen;
  const int tlen = d->tlen;
  const int m = d->m;
  const int o_del = d->o_del; 
  const int e_del = d->e_del; 
  const int o_ins = d->o_ins; 
  const int e_ins = d->e_ins; 
  const int w = d->w;
  const int end_bonus = d->end_bonus;
  const int zdrop = d->zdrop;
  const int h0 = d->h0;


  unsigned char *d_query;
 d_query = (unsigned char *)sycl::malloc_device(qlen, q_ct1);
 q_ct1.memcpy(d_query, d->query, qlen);

  unsigned char *d_target;
 d_target = (unsigned char *)sycl::malloc_device(tlen, q_ct1);
 q_ct1.memcpy(d_target, d->target, tlen);

  char *d_mat;
 d_mat = (char *)sycl::malloc_device(m * m, q_ct1);
 q_ct1.memcpy(d_mat, d->mat, m * m);

  eh_t *d_eh;
 d_eh = sycl::malloc_device<eh_t>((qlen + 1), q_ct1);
 q_ct1.memcpy(d_eh, eh, (qlen + 1) * sizeof(eh_t));

  char *d_qp;
 d_qp = (char *)sycl::malloc_device(qlen * m, q_ct1);
 q_ct1.memcpy(d_qp, qp, qlen * m);

  int *d_qle;
 d_qle = (int *)sycl::malloc_device(4, q_ct1);

  int *d_tle;
 d_tle = (int *)sycl::malloc_device(4, q_ct1);

  int *d_gtle;
 d_gtle = (int *)sycl::malloc_device(4, q_ct1);

  int *d_gscore;
 d_gscore = (int *)sycl::malloc_device(4, q_ct1);

  int *d_max_off;
 d_max_off = (int *)sycl::malloc_device(4, q_ct1);

  int *d_score;
 d_score = (int *)sycl::malloc_device(4, q_ct1);

 q_ct1.submit([&](sycl::handler &cgh) {
  cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
       kernel_extend2(d_query, d_target, d_mat, d_eh, d_qp, d_qle, d_tle,
                      d_gtle, d_gscore, d_max_off, d_score, qlen, tlen, m,
                      o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0);
      });
 });

 q_ct1.memcpy(&qle, d_qle, 4).wait();
 q_ct1.memcpy(&tle, d_tle, 4).wait();
 q_ct1.memcpy(&gtle, d_gtle, 4).wait();
 q_ct1.memcpy(&max_off, d_max_off, 4).wait();
 q_ct1.memcpy(&gscore, d_gscore, 4).wait();
 q_ct1.memcpy(&score, d_score, 4).wait();

 sycl::free(d_query, q_ct1);
 sycl::free(d_target, q_ct1);
 sycl::free(d_mat, q_ct1);
 sycl::free(d_eh, q_ct1);
 sycl::free(d_qp, q_ct1);
 sycl::free(d_qle, q_ct1);
 sycl::free(d_tle, q_ct1);
 sycl::free(d_gtle, q_ct1);
 sycl::free(d_gscore, q_ct1);
 sycl::free(d_max_off, q_ct1);
 sycl::free(d_score, q_ct1);

  check(d->qle, qle, "qle");
  check(d->tle, tle, "tle");
  check(d->gtle, gtle, "gtle");
  check(d->gscore, gscore, "gscore");
  check(d->max_off, max_off, "max_off");
  check(d->score, score, "score");

  free(eh);
  free(qp);

#ifdef VERBOSE
  printf("device: qle=%d, tle=%d, gtle=%d, gscore=%d, max_off=%d, score=%d\n",
      qle, tle, gtle, gscore, max_off, score);
#endif
}

int main(int argc, char *argv[])
{
  int iterations = atoi(argv[1]);

  struct extend2_dat d;

  // Instead of iterating over a directory, list the file names (17 in total)
  const char* files[] = {
#include "filelist.txt"
  };

  for (int f = 0; f < iterations; f++) {
    read_data(files[f%17], &d);
    extend2(&d);
  }
  return 0;
}
