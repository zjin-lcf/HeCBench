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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "common.h"
#include "read_data.h"

static void check(int a, int b, const char *s)
{
  if (a != b) printf("Error: %s %d %d\n", s, a, b);
}

typedef struct {
  int h, e;
} eh_t;


void extend2(queue &q, struct extend2_dat *d)
{
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
  int w = d->w;
  const int end_bonus = d->end_bonus;
  const int zdrop = d->zdrop;
  const int h0 = d->h0;

  {

    buffer<unsigned char, 1> d_query (d->query, qlen);
    buffer<unsigned char, 1> d_target (d->target, tlen);
    buffer<char, 1> d_mat (d->mat, m*m);
    buffer<int, 1> d_qle (&qle, 1);
    buffer<int, 1> d_tle (&tle, 1);
    buffer<int, 1> d_gtle (&gtle, 1);
    buffer<int, 1> d_gscore (&gscore, 1);
    buffer<int, 1> d_max_off (&max_off, 1);
    buffer<int, 1> d_score (&score, 1);
    buffer<eh_t, 1> d_eh (eh, qlen + 1);
    buffer<char, 1> d_qp (qp, qlen*m);
    d_eh.set_final_data(nullptr);
    d_qp.set_final_data(nullptr);

    q.submit([&](handler &h) {
        auto query = d_query.get_access<sycl_read>(h);
        auto target = d_target.get_access<sycl_read>(h);
        auto mat = d_mat.get_access<sycl_read>(h);
        auto eh = d_eh.get_access<sycl_write>(h);
        auto qp = d_qp.get_access<sycl_write>(h);
        auto qle_acc = d_qle.get_access<sycl_discard_write>(h);
        auto tle_acc = d_tle.get_access<sycl_discard_write>(h);
        auto gtle_acc = d_gtle.get_access<sycl_discard_write>(h);
        auto gscore_acc = d_gscore.get_access<sycl_discard_write>(h);
        auto max_off_acc = d_max_off.get_access<sycl_discard_write>(h);
        auto score_acc = d_score.get_access<sycl_discard_write>(h);

        h.single_task<class ebwa>([=]() {
          int oe_del = o_del + e_del;
          int oe_ins = o_ins + e_ins; 
          int i, j, k;
          int beg, end;
          int max, max_i, max_j, max_ins, max_del, max_ie;
          int max_w; // w is not mutable by default
          int gscore;
          int max_off;
	  int abs_v;

          // generate the query profile
          for (k = i = 0; k < m; ++k) {
            char *p = mat.get_pointer() + k * m;
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
          max_ins = (int)((float)(qlen * max + end_bonus - o_ins) / e_ins + 1.f);
          max_ins = max_ins > 1? max_ins : 1;
          max_w = w < max_ins? w : max_ins;
          max_del = (int)((float)(qlen * max + end_bonus - o_del) / e_del + 1.f);
          max_del = max_del > 1? max_del : 1;
          max_w = max_w < max_del? max_w : max_del; // TODO: is this necessary?
          // DP loop
          max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1;
          max_off = 0;
          beg = 0, end = qlen;
          for (i = 0; i < tlen; ++i) {
            int t, f = 0, h1, m = 0, mj = -1;
            char *q = qp.get_pointer() + target[i] * qlen;

            // apply the band and the constraint (if provided)
            if (beg < i - max_w) beg = i - max_w;
            if (end > i + max_w + 1) end = i + max_w + 1;
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
              eh_t *p = eh.get_pointer()+j;
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
	      abs_v = ( mj -i ) < 0 ? i - mj : mj - i;
              max_off = max_off > abs_v ? max_off : abs_v;
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
          qle_acc[0] = max_j + 1;
          tle_acc[0] = max_i + 1;
          gtle_acc[0] = max_ie + 1;
          gscore_acc[0] = gscore;
          max_off_acc[0] = max_off;
          score_acc[0] = max;
        });
    });
  }

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

#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  struct extend2_dat d;

  // list the file names (17 in total)
  const char* files[] = {
#include "filelist.txt"
  };

  for (int f = 0; f < iterations; f++) {
    read_data(files[f%17], &d);
    extend2(q, &d);
  }
  return 0;
}
