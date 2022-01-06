/* 
   FSM_GA is a GPU-accelerated implementation of a genetic algorithm
   (GA) for finding well-performing finite-state machines (FSM) for predicting
   binary sequences.

   Copyright (c) 2013, Texas State University. All rights reserved.

   Redistribution and use in source and binary forms, with or without modification,
   are permitted for academic, research, experimental, or personal use provided
   that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions, and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions, and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.

 For all other uses, please contact the Office for Commercialization and Industry
 Relations at Texas State University <http://www.txstate.edu/ocir/>.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include "common.h"
#include "parameters.h"
#include "kernels.h"

int main(int argc, char *argv[])
{
  int i, j, d, s, bit, length, pc, misses, besthits, generations;
  unsigned short *data;
  unsigned char state[TABSIZE], fsm[FSMSIZE * 2];
  int best[FSMSIZE * 2 + 3], trans[FSMSIZE][2];
  double runtime;
  struct timeval starttime, endtime;

  if (argc != 2) {fprintf(stderr, "usage: %s trace_length\n", argv[0]); exit(-1);}

  length = atoi(argv[1]);

  assert(sizeof(unsigned short) == 2);
  assert(0 < length);
  assert((FSMSIZE & (FSMSIZE - 1)) == 0);
  assert((TABSIZE & (TABSIZE - 1)) == 0);
  assert((0 < FSMSIZE) && (FSMSIZE <= 256));
  assert((0 < TABSIZE) && (TABSIZE <= 32768));
  assert(0 < POPCNT);
  assert((0 < POPSIZE) && (POPSIZE <= 1024));
  assert(0 < CUTOFF);

  data = (unsigned short*) malloc (sizeof(unsigned short) * length);

  srand(123);
  for (int i = 0; i < length; i++) data[i] = rand();

  printf("%d\t#kernel execution times\n", REPEAT);
  printf("%d\t#fsm size\n", FSMSIZE);
  printf("%d\t#entries\n", length);
  printf("%d\t#tab size\n", TABSIZE);
  printf("%d\t#blocks\n", POPCNT);
  printf("%d\t#threads\n", POPSIZE);
  printf("%d\t#cutoff\n", CUTOFF);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<unsigned short, 1> d_data (data, length);
  buffer<int, 1> d_best (FSMSIZE*2+3);
  buffer<unsigned int, 1> d_state (POPCNT * POPSIZE);
  buffer<unsigned char, 1> d_bfsm (POPCNT * FSMSIZE *2);
  buffer<unsigned char, 1> d_same (POPCNT);
  buffer<int, 1> d_smax (POPCNT);
  buffer<int, 1> d_sbest (POPCNT);
  buffer<int, 1> d_oldmax (POPCNT);

  gettimeofday(&starttime, NULL);
  for (int i = 0; i < REPEAT; i++) {
    q.submit([&](sycl::handler& cgh) {
      auto acc = d_best.get_access<sycl_write>(cgh);
      cgh.fill(acc, 0);
    });

    range<1> lws (POPSIZE);
    range<1> gws (POPCNT * POPSIZE);

    q.submit([&](handler& cgh) {
      auto data = d_data.get_access<sycl_read>(cgh);
      auto best = d_best.get_access<sycl_read_write>(cgh);
      auto state = d_state.get_access<sycl_read_write>(cgh);
      auto bfsm = d_bfsm.get_access<sycl_write>(cgh);
      auto same = d_same.get_access<sycl_read_write>(cgh);
      auto smax = d_smax.get_access<sycl_read_write>(cgh);
      auto sbest = d_sbest.get_access<sycl_read_write>(cgh);
      auto oldmax = d_oldmax.get_access<sycl_read_write>(cgh);
      accessor<unsigned char, 1, sycl_read_write, access::target::local> sm (FSMSIZE*2*POPSIZE, cgh);
      cgh.parallel_for<class fsm>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        FSMKernel(item, length,
                  data.get_pointer(),
                  best.get_pointer(),
                  state.get_pointer(), 
                  bfsm.get_pointer(),
                  same.get_pointer(),
                  smax.get_pointer(),
                  sbest.get_pointer(),
                  oldmax.get_pointer(),
                  sm.get_pointer());
      });
    });

    q.submit([&](handler& cgh) {
      auto best = d_best.get_access<sycl_write>(cgh);
      auto bfsm = d_bfsm.get_access<sycl_read>(cgh);
      cgh.single_task([=] () {
         MaxKernel(best.get_pointer(), bfsm.get_pointer()); 
      });
    });
  }
  q.wait();

  gettimeofday(&endtime, NULL);

  q.submit([&](sycl::handler& cgh) {
    auto acc = d_best.get_access<sycl_read>(cgh);
    cgh.copy(acc, best);
  }).wait();

  besthits = best[1];
  generations = best[2];

  runtime = endtime.tv_sec + endtime.tv_usec / 1000000.0 - starttime.tv_sec - starttime.tv_usec / 1000000.0;
  printf("%.6lf\t#runtime [s]\n", runtime / REPEAT);
  printf("%.6lf\t#throughput [Gtr/s]\n", 0.000000001 * POPSIZE * generations * length / (runtime / REPEAT));

  // evaluate saturating up/down counter
  for (i = 0; i < FSMSIZE; i++) {
    fsm[i * 2 + 0] = i - 1;
    fsm[i * 2 + 1] = i + 1;
  }
  fsm[0] = 0;
  fsm[(FSMSIZE - 1) * 2 + 1] = FSMSIZE - 1;
  memset(state, 0, TABSIZE);
  misses = 0;
  for (i = 0; i < length; i++) {
    d = (int)data[i];
    pc = (d >> 1) & (TABSIZE - 1);
    bit = d & 1;
    s = (int)state[pc];
    misses += bit ^ (((s + s) / FSMSIZE) & 1);
    state[pc] = fsm[s + s + bit];
  }
  printf("%d\t#sudcnt hits\n", length-misses);
  printf("%d\t#GAfsm hits\n", besthits);

  printf("%.3lf%%\t#sudcnt hits\n", 100.0 * (length - misses) / length);
  printf("%.3lf%%\t#GAfsm hits\n\n", 100.0 * besthits / length);

  // verify result and count transitions
  for (i = 0; i < FSMSIZE; i++) {
    for (j = 0; j < 2; j++) {
      trans[i][j] = 0;
    }
  }
  for (i = 0; i < FSMSIZE * 2; i++) {
    fsm[i] = best[i + 3];
  }
  memset(state, 0, TABSIZE);
  misses = 0;
  for (i = 0; i < length; i++) {
    d = (int)data[i];
    pc = (d >> 1) & (TABSIZE - 1);
    bit = d & 1;
    s = (int)state[pc];
    trans[s][bit]++;
    misses += bit ^ (s & 1);
    state[pc] = (unsigned char)fsm[s + s + bit];
  }

  bool ok = ((length - misses) == besthits);
  printf("%s\n", ok ? "PASS" : "FAIL");
  
#ifdef DEBUG
  // print FSM state assignment in R's ncol format
  for (bit = 0; bit < 2; bit++) {
    for (s = 0; s < FSMSIZE; s++) {
      d = fsm[s + s + bit];
      printf("%c%d %c%d %d\n", (s & 1) ? 'P' : 'N', s / 2, (d & 1) ? 'P' : 'N', d / 2, ((bit * 2) - 1) * trans[s][bit]);
    }
  }
#endif

  free(data);
  return 0;
}
