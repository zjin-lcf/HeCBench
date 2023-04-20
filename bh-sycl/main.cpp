/*
   ECL-BH v4.5: Simulation of the gravitational forces in a star cluster using
   the Barnes-Hut n-body algorithm.

   Copyright (c) 2010-2020 Texas State University. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher and Sahar Azimi

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/.

Publication: This work is described in detail in the following paper.
Martin Burtscher and Keshav Pingali. An Efficient CUDA Implementation of the
Tree-based Barnes Hut n-Body Algorithm. Chapter 6 in GPU Computing Gems
Emerald Edition, pp. 75-92. January 2011.
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <sycl/sycl.hpp>

// threads per block
#define THREADS1 256  // must be a power of 2
#define THREADS2 256
#define THREADS3 256 
#define THREADS4 256
#define THREADS5 256
#define THREADS6 256

// block count = factor * #SMs
#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  // must all be resident at the same time
#define FACTOR4 1  // must all be resident at the same time
#define FACTOR5 2
#define FACTOR6 2

#define WARPSIZE 32
#define MAXDEPTH 32


// random number generator (https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/util.C)

static int randx = 7;
static double drnd()
{
  const int lastrand = randx;
  randx = (1103515245L * randx + 12345) & 0x7FFFFFFF;
  return (double)lastrand / 2147483648.0;
}

// Reference
inline unsigned int atomicInc(unsigned int *addr,
                              unsigned int operand) 
{
  auto atm = sycl::atomic_ref<unsigned int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(addr[0]);
  unsigned int old;
  while (true) {
    old = atm.load();
    if (old >= operand) {
      if (atm.compare_exchange_strong(old, 0))
        break;
    } else if (atm.compare_exchange_strong(old, old + 1))
      break;
  }
  return old;
}

inline int atomicCAS(int &val, int expected, int desired) 
{
  int expected_value = expected;
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

inline int atomicSub(int &val, int operand) 
{
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  return atm.fetch_sub(operand);
}


int main(int argc, char* argv[])
{
  // perform some checks

  printf("ECL-BH v4.5\n");
  printf("Copyright (c) 2010-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps\n");
    exit(-1);
  }

  const int nbodies = atoi(argv[1]);
  if (nbodies < 1) {
    fprintf(stderr, "nbodies is too small: %d\n", nbodies);
    exit(-1);
  }

  if (nbodies > (1 << 30)) {
    fprintf(stderr, "nbodies is too large: %d\n", nbodies);
    exit(-1);
  }

  int timesteps = atoi(argv[2]);
  if (timesteps < 0) {
    fprintf(stderr, "the number of steps should be positive: %d\n", timesteps);
    exit(-1);
  }

  int i;
  int nnodes, step;
  double runtime;
  float dtime, dthf, epssq, itolsq;

  sycl::float4 *accVel;
  sycl::float2 *vel;
  sycl::float4 *posMass;
  double rsc, vsc, r, v, x, y, z, sq, scale;

  // the number of thread blocks may be adjusted for higher performance
  const int blocks = 24;

  nnodes = nbodies * 2;
  if (nnodes < 1024*blocks) nnodes = 1024*blocks;
  while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
  nnodes--;

  dtime = 0.025f; 
  dthf = dtime * 0.5f;
  epssq = 0.05 * 0.05;
  itolsq = 1.0f / (0.5 * 0.5);

  printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

  // allocate host memory

  accVel = (sycl::float4*)malloc(sizeof(sycl::float4) * nbodies);
  if (accVel == NULL) fprintf(stderr, "cannot allocate accVel\n");

  vel = (sycl::float2*)malloc(sizeof(sycl::float2) * nbodies);
  if (vel == NULL) fprintf(stderr, "cannot allocate vel\n");

  posMass = (sycl::float4*)malloc(sizeof(sycl::float4) * nbodies);
  if (posMass == NULL) fprintf(stderr, "cannot allocate posMass\n");

  // initialize host memory (https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/code.C)

  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < nbodies; i++) {
    sycl::float4 p;
    p.w() = 1.f / nbodies;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1.0);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    p.x() = x * scale;
    p.y() = y * scale;
    p.z() = z * scale;
    posMass[i] = p;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    sycl::float2 v;
    v.x() = x * scale;
    v.y() = y * scale;
    accVel[i].w() = z * scale;
    vel[i] = v;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // allocate device memory

  int *childd = sycl::malloc_device<int>((nnodes+1) * 8, q);
  int *countd = sycl::malloc_device<int>(nnodes+1, q);
  int *startd = sycl::malloc_device<int>(nnodes+1, q);
  int *sortd = sycl::malloc_device<int>(nnodes+1, q);

  sycl::float2 *veld = sycl::malloc_device<sycl::float2>(nnodes+1, q);
  sycl::float4 *accVeld = sycl::malloc_device<sycl::float4>(nnodes+1, q);
  sycl::float4 *posMassd = sycl::malloc_device<sycl::float4>(nnodes+1, q);
  sycl::float3 *maxd = sycl::malloc_device<sycl::float3>(blocks * FACTOR1, q);
  sycl::float3 *mind = sycl::malloc_device<sycl::float3>(blocks * FACTOR1, q);
  int *stepd = sycl::malloc_device<int>(1, q);
  int *bottomd = sycl::malloc_device<int>(1, q);
  unsigned int *blkcntd = sycl::malloc_device<unsigned int>(1, q);
  float *radiusd = sycl::malloc_device<float>(1, q);

  q.memcpy(accVeld, accVel, nbodies * sizeof(sycl::float4));
  q.memcpy(veld, vel, nbodies * sizeof(sycl::float2));
  q.memcpy(posMassd, posMass, nbodies * sizeof(sycl::float4));

  q.wait();

  struct timeval starttime, endtime;
  gettimeofday(&starttime, NULL);

  // run timesteps (launch kernels on a device)
  q.submit([&] (sycl::handler &cgh) {
    cgh.single_task<class init>([=] () {
      stepd[0] = -1;
      blkcntd[0] = 0;
    });
  });

  for (step = 0; step < timesteps; step++) {

    sycl::range<1> k2_gws (blocks * FACTOR1 * THREADS1);
    sycl::range<1> k2_lws (THREADS1);

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> smaxx(sycl::range<1>(THREADS1), cgh);
      sycl::local_accessor<float, 1> smaxy(sycl::range<1>(THREADS1), cgh);
      sycl::local_accessor<float, 1> smaxz(sycl::range<1>(THREADS1), cgh);
      sycl::local_accessor<float, 1> sminx(sycl::range<1>(THREADS1), cgh);
      sycl::local_accessor<float, 1> sminy(sycl::range<1>(THREADS1), cgh);
      sycl::local_accessor<float, 1> sminz(sycl::range<1>(THREADS1), cgh);
      cgh.parallel_for<class bounding_box>(
        sycl::nd_range<1>(k2_gws, k2_lws), [=] (sycl::nd_item<1> item) {
        int i, j, k;
        float val;
        sycl::float3 min, max;
        // initialize with valid data (in case #bodies < #threads)
        const sycl::float4 p0 = posMassd[0];
        min.x() = max.x() = p0.x();
        min.y() = max.y() = p0.y();
        min.z() = max.z() = p0.z();

        // scan all bodies
        i = item.get_local_id(0);
        int inc = THREADS1 * item.get_group_range(0);
        for (j = item.get_global_id(0); j < nbodies; j += inc) {
          const sycl::float4 p = posMassd[j];
          val = p.x();
          min.x() = sycl::fmin(min.x(), val);
          max.x() = sycl::fmax(max.x(), val);
          val = p.y();
          min.y() = sycl::fmin(min.y(), val);
          max.y() = sycl::fmax(max.y(), val);
          val = p.z();
          min.z() = sycl::fmin(min.z(), val);
          max.z() = sycl::fmax(max.z(), val);
        }

        // reduction in shared memory
        sminx[i] = min.x();
        smaxx[i] = max.x();
        sminy[i] = min.y();
        smaxy[i] = max.y();
        sminz[i] = min.z();
        smaxz[i] = max.z();

        for (j = THREADS1 / 2; j > 0; j /= 2) {
          item.barrier(sycl::access::fence_space::local_space);
          if (i < j) {
            k = i + j;
            sminx[i] = min.x() = sycl::fmin(min.x(), sminx[k]);
            smaxx[i] = max.x() = sycl::fmax(max.x(), smaxx[k]);
            sminy[i] = min.y() = sycl::fmin(min.y(), sminy[k]);
            smaxy[i] = max.y() = sycl::fmax(max.y(), smaxy[k]);
            sminz[i] = min.z() = sycl::fmin(min.z(), sminz[k]);
            smaxz[i] = max.z() = sycl::fmax(max.z(), smaxz[k]);
          }
        }

        // write block result to global memory
        if (i == 0) {
          k = item.get_group(0);
          mind[k] = min;
          maxd[k] = max;

          sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

          inc = item.get_group_range(0) - 1;
          
          if (inc == atomicInc(blkcntd, inc)) {
            // I'm the last block, so combine all block results
            for (j = 0; j <= inc; j++) {
              sycl::float3 minp = mind[j];
              sycl::float3 maxp = maxd[j];
              min.x() = sycl::fmin(min.x(), minp.x());
              max.x() = sycl::fmax(max.x(), maxp.x());
              min.y() = sycl::fmin(min.y(), minp.y());
              max.y() = sycl::fmax(max.y(), maxp.y());
              min.z() = sycl::fmin(min.z(), minp.z());
              max.z() = sycl::fmax(max.z(), maxp.z());
            }

            // compute radius
            val = sycl::fmax(max.x() - min.x(), max.y() - min.y());
            radiusd[0] = sycl::fmax(val, max.z() - min.z()) * 0.5f;

            // create root node
            k = nnodes;
            bottomd[0] = k;

            startd[k] = 0;
            sycl::float4 p;
            p.x() = (min.x() + max.x()) * 0.5f;
            p.y() = (min.y() + max.y()) * 0.5f;
            p.z() = (min.z() + max.z()) * 0.5f;
            p.w() = -1.0f;
            posMassd[k] = p;
            k *= 8;
            for (i = 0; i < 8; i++) childd[k + i] = -1;
            stepd[0]++;
          }
        }
      });
    });

    sycl::range<1> k3_gws (blocks * 256);
    sycl::range<1> k3_lws (256);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class clear_kernel1>(
        sycl::nd_range<1>(k3_gws, k3_lws), [=] (sycl::nd_item<1> item) {
        int top = 8 * nnodes;
        int bottom = 8 * nbodies;
        int inc = item.get_local_range(0) * item.get_group_range(0);
        int k = (bottom & (-WARPSIZE)) + item.get_global_id(0);  // align to warp size
        if (k < bottom) k += inc;

        // iterate over all cells assigned to thread
        while (k < top) {
          childd[k] = -1;
          k += inc;
        }
      });
    });

    sycl::range<1> k4_gws (blocks * FACTOR2 * THREADS2);
    sycl::range<1> k4_lws (THREADS2);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class tree_building>(sycl::nd_range<1>(k4_gws, k4_lws), [=] (sycl::nd_item<1> item) {
        int i, j, depth, skip, inc;
        float x, y, z, r;
        float dx, dy, dz;
        int ch, n, cell, locked, patch;
        float radius;

        // cache root data
        radius = radiusd[0] * 0.5f;
        const sycl::float4 root = posMassd[nnodes];

        skip = 1;
        inc = item.get_local_range(0) * item.get_group_range(0);
        i = item.get_global_id(0);

        // iterate over all bodies assigned to thread
        while (i < nbodies) {
          const sycl::float4 p = posMassd[i];
          if (skip != 0) {
            // new body, so start traversing at root
            skip = 0;
            n = nnodes;
            depth = 1;
            r = radius;
            dx = dy = dz = -r;
            j = 0;
            // determine which child to follow
            if (root.x() < p.x()) {j = 1; dx = r;}
            if (root.y() < p.y()) {j |= 2; dy = r;}
            if (root.z() < p.z()) {j |= 4; dz = r;}
            x = root.x() + dx;
            y = root.y() + dy;
            z = root.z() + dz;
          }

          // follow path to leaf cell
          ch = childd[n*8+j];
          while (ch >= nbodies) {
            n = ch;
            depth++;
            r *= 0.5f;
            dx = dy = dz = -r;
            j = 0;
            // determine which child to follow
            if (x < p.x()) {j = 1; dx = r;}
            if (y < p.y()) {j |= 2; dy = r;}
            if (z < p.z()) {j |= 4; dz = r;}
            x += dx;
            y += dy;
            z += dz;
            ch = childd[n*8+j];
          }

          if (ch != -2) {  // skip if child pointer is locked and try again later
            locked = n*8+j;
            if (ch == -1) {
              if (ch == atomicCAS(childd[locked], ch, i)) {  // if null, just insert the new body
                i += inc;  // move on to next body
                skip = 1;
              }
            } else {  // there already is a body at this position
              if (ch == atomicCAS(childd[locked], ch, -2)) {  // try to lock
                patch = -1;
                const sycl::float4 chp = posMassd[ch];
                // create new cell(s) and insert the old and new bodies
                do {
                  depth++;
                  cell = atomicSub(bottomd[0], 1) - 1;

                  if (patch != -1) {
                    childd[n*8+j] = cell;
                  }
                  patch = sycl::max(patch, cell);

                  j = 0;
                  if (x < chp.x()) j = 1;
                  if (y < chp.y()) j |= 2;
                  if (z < chp.z()) j |= 4;
                  childd[cell*8+j] = ch;

                  n = cell;
                  r *= 0.5f;
                  dx = dy = dz = -r;
                  j = 0;
                  if (x < p.x()) {j = 1; dx = r;}
                  if (y < p.y()) {j |= 2; dy = r;}
                  if (z < p.z()) {j |= 4; dz = r;}
                  x += dx;
                  y += dy;
                  z += dz;

                  ch = childd[n*8+j];
                  // repeat until the two bodies are different children
                } while (ch >= 0);
                childd[n*8+j] = i;

                i += inc;  // move on to next body
                skip = 2;
              }
            }
          }
          item.barrier(sycl::access::fence_space::local_space);  // optional barrier for performance

          if (skip == 2) {
            childd[locked] = patch;
          }
        }
      });
    });

    sycl::range<1> k5_gws (blocks * 256);
    sycl::range<1> k5_lws (256);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class clear_kernel2>(
        sycl::nd_range<1>(k5_gws, k5_lws), [=] (sycl::nd_item<1> item) {
        int bottom = bottomd[0];
        int inc = item.get_local_range(0) * item.get_group_range(0);
        int k = (bottom & (-WARPSIZE)) + item.get_global_id(0);  // align to warp size
        if (k < bottom) k += inc;

        // iterate over all cells assigned to thread
        while (k < nnodes) {
          posMassd[k].w() = -1.0f;
          startd[k] = -1;
          k += inc;
        }
      });
    });

    sycl::range<1> k6_gws (blocks * FACTOR3 * THREADS3);
    sycl::range<1> k6_lws (THREADS3);
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> child(sycl::range<1>(THREADS3*8), cgh);
      sycl::local_accessor<float, 1> mass(sycl::range<1>(THREADS3*8), cgh);
      cgh.parallel_for<class sum>(
        sycl::nd_range<1>(k6_gws, k6_lws), [=] (sycl::nd_item<1> item) {
        int i, j, ch, cnt;
        float cm, px, py, pz, m;
        int bottom = bottomd[0];
        int lid = item.get_local_id(0);
        int inc = item.get_local_range(0) * item.get_group_range(0);
        int k = (bottom & (-WARPSIZE)) + item.get_global_id(0);  // align to warp size
        if (k < bottom) k += inc;
      
        int restart = k;
        for (j = 0; j < 3; j++) {  // wait-free pre-passes
          // iterate over all cells assigned to thread
          while (k <= nnodes) {
            if (posMassd[k].w() < 0.0f) {
              for (i = 0; i < 8; i++) {
                ch = childd[k*8+i];
                child[i*THREADS3+lid] = ch;  // cache children
                if ((ch >= nbodies) && ((mass[i*THREADS3+lid] = posMassd[ch].w()) < 0.0f)) {
                  break;
                }
              }
              if (i == 8) {
                // all children are ready
                cm = 0.0f;
                px = 0.0f;
                py = 0.0f;
                pz = 0.0f;
                cnt = 0;
                for (i = 0; i < 8; i++) {
                  ch = child[i*THREADS3+lid];
                  if (ch >= 0) {
                    const float chx = posMassd[ch].x();
                    const float chy = posMassd[ch].y();
                    const float chz = posMassd[ch].z();
                    const float chw = posMassd[ch].w();
                    if (ch >= nbodies) {  // count bodies (needed later)
                      m = mass[i*THREADS3+lid];
                      cnt += countd[ch];
                    } else {
                      m = chw;
                      cnt++;
                    }
                    // add child's contribution
                    cm += m;
                    px += chx * m;
                    py += chy * m;
                    pz += chz * m;
                  }
                }
                countd[k] = cnt;
                m = 1.0f / cm;
                posMassd[k].x() = px * m;
                posMassd[k].y() = py * m;
                posMassd[k].z() = pz * m;
                posMassd[k].w() = cm;
              }
            }
            k += inc;  // move on to next cell
          }
          k = restart;
        }
      
        j = 0;
        // iterate over all cells assigned to thread
        while (k <= nnodes) {
          if (posMassd[k].w() >= 0.0f) {
            k += inc;
          } else {
            if (j == 0) {
              j = 8;
              for (i = 0; i < 8; i++) {
                ch = childd[k*8+i];
                child[i*THREADS3+lid] = ch;  // cache children
                if ((ch < nbodies) || ((mass[i*THREADS3+lid] = posMassd[ch].w()) >= 0.0f)) {
                  j--;
                }
              }
            } else {
              j = 8;
              for (i = 0; i < 8; i++) {
                ch = child[i*THREADS3+lid];
                if ((ch < nbodies) || (mass[i*THREADS3+lid] >= 0.0f) || ((mass[i*THREADS3+lid] = posMassd[ch].w()) >= 0.0f)) {
                  j--;
                }
              }
            }
      
            if (j == 0) {
              // all children are ready
              cm = 0.0f;
              px = 0.0f;
              py = 0.0f;
              pz = 0.0f;
              cnt = 0;
              for (i = 0; i < 8; i++) {
                ch = child[i*THREADS3+lid];
                if (ch >= 0) {
                  const float chx = posMassd[ch].x();
                  const float chy = posMassd[ch].y();
                  const float chz = posMassd[ch].z();
                  const float chw = posMassd[ch].w();
                  if (ch >= nbodies) {  // count bodies (needed later)
                    m = mass[i*THREADS3+lid];
                    cnt += countd[ch];
                  } else {
                    m = chw;
                    cnt++;
                  }
                  // add child's contribution
                  cm += m;
                  px += chx * m;
                  py += chy * m;
                  pz += chz * m;
                }
              }
              countd[k] = cnt;
              m = 1.0f / cm;
              posMassd[k].x() = px * m;
              posMassd[k].y() = py * m;
              posMassd[k].z() = pz * m;
              posMassd[k].w() = cm;
              k += inc;
            }
          }
        }
      });
    });

    sycl::range<1> k7_gws (blocks * FACTOR4 * THREADS4);
    sycl::range<1> k7_lws (THREADS4);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class sort>(
        sycl::nd_range<1>(k7_gws, k7_lws), [=] (sycl::nd_item<1> item) {
        int i, j;
        int bottom = bottomd[0];
        int dec = item.get_local_range(0) * item.get_group_range(0);
        int k = nnodes + 1 - dec + item.get_global_id(0);

        // iterate over all cells assigned to thread
        while (k >= bottom) {
          int start = startd[k];
          if (start >= 0) {
            j = 0;
            for (i = 0; i < 8; i++) {
              int ch = childd[k*8+i];
              if (ch >= 0) {
                if (i != j) {
                  // move children to front (needed later for speed)
                  childd[k*8+i] = -1;
                  childd[k*8+j] = ch;
                }
                j++;
                if (ch >= nbodies) {
                  // child is a cell
                  startd[ch] = start;  // set start ID of child
                  start += countd[ch];  // add #bodies in subtree
                } else {
                  // child is a body
                  sortd[start] = ch;  // record body in 'sorted' array
                  start++;
                }
              }
            }
            k -= dec;  // move on to next cell
          }
          item.barrier(sycl::access::fence_space::local_space);  // optional barrier for performance
        }
      });
    });

    sycl::range<1> k8_gws (blocks * FACTOR5 * THREADS5);
    sycl::range<1> k8_lws (THREADS5);

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> pos(sycl::range<1>(THREADS5), cgh);
      sycl::local_accessor<int, 1> node(sycl::range<1>(THREADS5), cgh);
      sycl::local_accessor<float, 1> dq(sycl::range<1>(THREADS5), cgh);
      cgh.parallel_for<class calc_force>(
        sycl::nd_range<1>(k8_gws, k8_lws), [=] (sycl::nd_item<1> item) {
        int i, j, k, n, depth, base, sbase, diff, pd, nd;
        float ax, ay, az, dx, dy, dz, tmp;
        int lid = item.get_local_id(0);
        if (0 == lid) {
          tmp = radiusd[0] * 2;
          // precompute values that depend only on tree level
          dq[0] = tmp * tmp * itolsq;
          for (i = 1; i < MAXDEPTH; i++) {
            dq[i] = dq[i - 1] * 0.25f;
            dq[i - 1] += epssq;
          }
          dq[i - 1] += epssq;
        }
        item.barrier(sycl::access::fence_space::local_space);
      
        // figure out first thread in each warp (lane 0)
        base = lid / WARPSIZE;
        sbase = base * WARPSIZE;
        j = base * MAXDEPTH;
      
        diff = lid - sbase;
        // make multiple copies to avoid index calculations later
        if (diff < MAXDEPTH) {
          dq[diff+j] = dq[diff];
        }
        item.barrier(sycl::access::fence_space::local_space);
      
        // iterate over all bodies assigned to thread
        for (k = item.get_global_id(0); k < nbodies; k += item.get_local_range(0) * item.get_group_range(0)) {
          i = sortd[k];  // get permuted/sorted index
          // cache position info
          const sycl::float4 pi = posMassd[i];
      
          ax = 0.0f;
          ay = 0.0f;
          az = 0.0f;
      
          // initialize iteration stack, i.e., push root node onto stack
          depth = j;
          if (sbase == lid) {
            pos[j] = 0;
            node[j] = nnodes * 8;
          }
      
          do {
            // stack is not empty
            pd = pos[depth];
            nd = node[depth];
            while (pd < 8) {
              // node on top of stack has more children to process
              n = childd[nd + pd];  // load child pointer
              pd++;
      
              if (n >= 0) {
                const sycl::float4 pn = posMassd[n];
                dx = pn.x() - pi.x();
                dy = pn.y() - pi.y();
                dz = pn.z() - pi.z();
                tmp = dx*dx + (dy*dy + (dz*dz + epssq));  // compute distance squared (plus softening)
                if ((n < nbodies) || sycl::all_of_group(item.get_group(), tmp >= dq[depth])) {  
                // check if all threads agree that cell is far enough away (or is a body)
                  tmp = sycl::rsqrt(tmp);  // compute distance
                  tmp = pn.w() * tmp * tmp * tmp;
                  ax += dx * tmp;
                  ay += dy * tmp;
                  az += dz * tmp;
                } else {
                  // push cell onto stack
                  if (sbase == lid) {
                    pos[depth] = pd;
                    node[depth] = nd;
                  }
                  depth++;
                  pd = 0;
                  nd = n * 8;
                }
              } else {
                pd = 8;  // early out because all remaining children are also zero
              }
            }
            depth--;  // done with this level
          } while (depth >= j);
      
          sycl::float4 acc = accVeld[i];
          if (stepd[0] > 0) {
            // update velocity
            sycl::float2 v = veld[i];
            v.x() += (ax - acc.x()) * dthf;
            v.y() += (ay - acc.y()) * dthf;
            acc.w() += (az - acc.z()) * dthf;
            veld[i] = v;
          }
      
          // save computed acceleration
          acc.x() = ax;
          acc.y() = ay;
          acc.z() = az;
          accVeld[i] = acc;
        }
      });
    });

    sycl::range<1> k9_gws (blocks * FACTOR6 * THREADS6);
    sycl::range<1> k9_lws (THREADS6);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class integration>(
        sycl::nd_range<1>(k9_gws, k9_lws), [=] (sycl::nd_item<1> item) {
        // iterate over all bodies assigned to thread
        int inc = item.get_local_range(0) * item.get_group_range(0);
        for (int i = item.get_global_id(0); i < nbodies; i += inc) {
          // integrate
          sycl::float4 acc = accVeld[i];
          float dvelx = acc.x() * dthf;
          float dvely = acc.y() * dthf;
          float dvelz = acc.z() * dthf;

          sycl::float2 v = veld[i];
          float velhx = v.x() + dvelx;
          float velhy = v.y() + dvely;
          float velhz = acc.w() + dvelz;

          sycl::float4 p = posMassd[i];
          p.x() += velhx * dtime;
          p.y() += velhy * dtime;
          p.z() += velhz * dtime;
          posMassd[i] = p;

          v.x() = velhx + dvelx;
          v.y() = velhy + dvely;
          acc.w() = velhz + dvelz;
          veld[i] = v;
          accVeld[i] = acc;
        }
      });
    });
  }
  q.wait();

  gettimeofday(&endtime, NULL);
  runtime = (endtime.tv_sec + endtime.tv_usec/1000000.0 - 
             starttime.tv_sec - starttime.tv_usec/1000000.0);

  printf("Total kernel execution time: %.4lf s\n", runtime);

  // transfer final results back to a host
  q.memcpy(accVel, accVeld, nbodies * sizeof(sycl::float4));
  q.memcpy(vel, veld, nbodies * sizeof(sycl::float2));
  q.memcpy(posMass, posMassd, nbodies * sizeof(sycl::float4));

  q.wait();

#ifdef DEBUG
  // print output for verification
  for (i = 0; i < nbodies; i++) {
    printf("%d: %.2e %.2e %.2e\n", i, posMass[i].x(), posMass[i].y(), posMass[i].z());
    printf("%d: %.2e %.2e %.2e %.2e\n", i, accVel[i].x(), accVel[i].y(), accVel[i].z(), accVel[i].w());
    printf("%d: %.2e %.2e\n", i, vel[i].x(), vel[i].y());
  }
#endif

  free(vel);
  free(accVel);
  free(posMass);

  sycl::free(childd, q);
  sycl::free(veld, q);
  sycl::free(accVeld, q);
  sycl::free(countd, q);
  sycl::free(startd, q);
  sycl::free(sortd, q);
  sycl::free(posMassd, q);
  sycl::free(maxd, q);
  sycl::free(mind, q);
  sycl::free(stepd, q);
  sycl::free(blkcntd, q);
  sycl::free(bottomd, q);
  sycl::free(radiusd, q);

  return 0;
}
