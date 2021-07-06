//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////

#ifndef _KERNEL_MONTECARLO_CUH_
#define _KERNEL_MONTECARLO_CUH_

// GPU-related definitions

#define sLx (BX+2)
#define sLy (BY+2)
#define sLz (BZ+2)

#define SVOLUME ((sLx)*(sLy)*(sLz))
#define BVOLUME ((BX)*((BY)/2)*(BZ))

#define BLOCKSIZE 8
#define BLOCK_STEPS 1
#define EPSILON 0.00000000001f

#define C(x,y,z,L)     ((z)*(L)*(L)+(y)*(L)+(x))
#define sC(x,y,z,Lx,Ly)  ((z+1)*(Ly)*(Lx)+(y+1)*(Lx)+(x+1))

void
kernel_metropolis(const int N, const int L, int *s, const int *H, 
                  const float h, const float B, uint64_t *state, uint64_t *inc, int alt)
{

  const int gridDim_x = (L + BX - 1) / BX;
  const int gridDim_y = (L + BY - 1) / (2*BY);
  const int gridDim_z = (L + BZ - 1) / BZ;

  const int numTeams = gridDim_x * gridDim_y * gridDim_z ;
  const int numThreads = BX * BY / 2 * BZ;

  #pragma omp target teams num_teams(numTeams) thread_limit(numThreads)
  {
    int ss[SVOLUME];
    #pragma omp parallel 
    {
      const int team = omp_get_team_num();
      const int thread = omp_get_thread_num();
      const int blockIdx_x = team % gridDim_x;
      const int blockIdx_y = team / gridDim_x % gridDim_y;
      const int blockIdx_z = team / gridDim_x / gridDim_y % gridDim_z;
      const int threadIdx_x = thread % BX;
      const int threadIdx_y = thread / BX % (BY/2);
      const int threadIdx_z = (thread * 2) / (BX * BY) % BZ;

      // offsets
      const int offx = blockIdx_x * BX;
      const int offy = (2*blockIdx_y + ((blockIdx_x + blockIdx_z + alt) & 1)) * BY;
      const int offz = blockIdx_z * BZ;

      // halo shared memory coords
      const int sx = threadIdx_x;
      const int sy1 = 2*threadIdx_y;
      const int sy2 = 2*threadIdx_y + 1;
      const int sz = threadIdx_z;

      // global coords
      const int x = offx + sx;
      const int y1 = offy + sy1;
      const int y2 = offy + sy2;
      const int z = offz + sz;

      //if(x >= N || y1 >= N || y2 >= N || z >= N)
      //return;

      // global and local and block id in soc
      const int tid = z*L*L/4 + (blockIdx_y * BY/2 + threadIdx_y)*L + x;

      // load the spins into shared memory
      ss[sC(sx, sy1, sz, sLx, sLy)] = s[C(x, y1, z, L)];
      ss[sC(sx, sy2, sz, sLx, sLy)] = s[C(x, y2, z, L)];
      // get the h1,h2 values for y1 y2_
      const int h1 = H[C(x, y1, z, L)];
      const int h2 = H[C(x, y2, z, L)];
      //printf("thread %i   h1=%i   h2=%i\n", tid, h1, h2);

      // ------------------------------------------------
      // halo
      // ------------------------------------------------
      // Y boundary
      if(threadIdx_y == 0){
        // we also check if we are on the limit of the lattice
        ss[sC(sx, -1, sz, sLx, sLy)] = (offy == 0) ? s[C(x, L-1, z, L)] : s[C(x, offy-1, z, L)];
      }
      if(threadIdx_y == BY/2-1){
        ss[sC(sx, BY, sz, sLx, sLy)] = (offy == L-BY) ? s[C(x, 0, z, L)] : s[C(x, offy+BY, z, L)];
      }

      // X boundary
      if(threadIdx_x == 0){
        if(blockIdx_x == 0){
          ss[sC(-1, sy1, sz, sLx, sLy)] = s[C(L-1, y1, z, L)];
          ss[sC(-1, sy2, sz, sLx, sLy)] = s[C(L-1, y2, z, L)];
        }
        else{
          ss[sC(-1, sy1, sz, sLx, sLy)] = s[C(offx-1, y1, z, L)];
          ss[sC(-1, sy2, sz, sLx, sLy)] = s[C(offx-1, y2, z, L)];
        }
      }
      if(threadIdx_x == BX-1){
        if(blockIdx_x == gridDim_x-1){
          ss[sC(BX, sy1, sz, sLx, sLy)] = s[C(0, y1, z, L)];
          ss[sC(BX, sy2, sz, sLx, sLy)] = s[C(0, y2, z, L)];
        }
        else{
          ss[sC(BX, sy1, sz, sLx, sLy)] = s[C(offx+BX, y1, z, L)];
          ss[sC(BX, sy2, sz, sLx, sLy)] = s[C(offx+BX, y2, z, L)];
        }
      }

      // Z boundary
      if(threadIdx_z == 0){
        if(blockIdx_z == 0){
          ss[sC(sx, sy1, -1, sLx, sLy)] = s[C(x, y1, L-1, L)];
          ss[sC(sx, sy2, -1, sLx, sLy)] = s[C(x, y2, L-1, L)];
        }
        else{
          ss[sC(sx, sy1, -1, sLx, sLy)] = s[C(x, y1, offz-1, L)];
          ss[sC(sx, sy2, -1, sLx, sLy)] = s[C(x, y2, offz-1, L)];
        }
      }
      if(threadIdx_z == BZ-1){
        if(blockIdx_z == gridDim_z-1){
          ss[sC(sx, sy1, BZ, sLx, sLy)] = s[C(x, y1, 0, L)];
          ss[sC(sx, sy2, BZ, sLx, sLy)] = s[C(x, y2, 0, L)];
        }
        else{
          ss[sC(sx, sy1, BZ, sLx, sLy)] = s[C(x, y1, offz+BZ, L)];
          ss[sC(sx, sy2, BZ, sLx, sLy)] = s[C(x, y2, offz+BZ, L)];
        }
      }

      // get random number state
      uint64_t lstate = state[tid];
      uint64_t linc = inc[tid];
      // the white and black y
      int wy = ((sx + sz) & 1)     + 2*threadIdx_y;
      int by = ((sx + sz + 1) & 1)  + 2*threadIdx_y;
      float dh;
      int c;

      #pragma omp barrier
      
      //#pragma unroll
      for(int i = 0; i < BLOCK_STEPS; ++i){

        /* -------- white update -------- */
        dh = (float)(ss[sC(sx, wy, sz, sLx, sLy)] * (
             (float)(ss[sC(sx-1,wy,sz, sLx, sLy)] + ss[sC(sx+1, wy, sz, sLx, sLy)] + 
                     ss[sC(sx,wy-1,sz, sLx, sLy)] + ss[sC(sx, wy+1, sz, sLx, sLy)] +
                     ss[sC(sx,wy,sz-1, sLx, sLy)] + ss[sC(sx, wy, sz+1, sLx, sLy)]) + h*h1));
        c = signbit(dh-EPSILON) | signbit(gpu_rand01(&lstate, &linc) - expf(dh * B));
        ss[sC(sx, wy, sz, sLx, sLy)] *= (1 - 2*c);
        #pragma omp barrier

        /* -------- black update -------- */
        dh = (float)(ss[sC(sx, by, sz, sLx, sLy)] * (
             (float)(ss[sC(sx-1,by,sz, sLx, sLy)] + ss[sC(sx+1, by, sz, sLx, sLy)] + 
                     ss[sC(sx,by-1,sz, sLx, sLy)] + ss[sC(sx, by+1, sz, sLx, sLy)] +
                     ss[sC(sx,by,sz-1, sLx, sLy)] + ss[sC(sx, by, sz+1, sLx, sLy)]) + h*h2));

        c = signbit(dh-EPSILON) | signbit(gpu_rand01(&lstate, &linc) - expf(dh * B));
        ss[sC(sx, by, sz, sLx, sLy)] *= (1 - 2*c);
        #pragma omp barrier
      }

      /* copy data back to gmem */
      s[C(x, y1, z, L)] = ss[sC(sx, sy1, sz, sLx, sLy)];
      s[C(x, y2, z, L)] = ss[sC(sx, sy2, sz, sLx, sLy)]; 
      /* update random number state */
      state[tid] = lstate;
      inc[tid] = linc;
    }
  }
}


// NOTE: the space of computation is 1/4 of N, so that is why each thread does quadruple work.
void kernel_reset_random_gpupcg(int *s, int N, uint64_t *state, uint64_t *inc)
{
  /* Each thread gets same seed, a different sequence number, no offset */

  #pragma omp target teams distribute parallel for thread_limit(BLOCKSIZE1D)
  for (int x = 0; x < N/4; x++) {

    /* get the prng state in register memory */
    uint64_t lstate = state[x];
    uint64_t linc = inc[x];

    // first random
    float v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
    s[x] = 1-2*v;
    // second random
    v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
    s[x + N/4] = 1-2*v;
    // third random
    v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
    s[x + N/2 ]  = 1-2*v;
    // fourth random
    v = (int) (gpu_rand01(&lstate, &linc) + 0.5f);
    s[x + 3*N/4] = 1-2*v;

    /* save the state back to global memory */
    state[x] = lstate;
    inc[x] = linc;
  }
}

template<typename T>
void kernel_reset(T *a, int N, T val){
  #pragma omp target teams distribute parallel for thread_limit(BLOCKSIZE1D)
  for (int idx = 0; idx < N; idx++) a[idx] = val;
}
#endif
