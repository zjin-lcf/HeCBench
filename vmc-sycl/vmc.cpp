#include <cstdio>
#include <cmath>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include "common.h"

using namespace std;

using FLOAT = float;

const int NTHR_PER_BLK = 256;           // Number of threads per block
const int NBLOCK  = 56*4;               // Number of blocks
const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Neq = 100000;                 // No. of generations to equilibrate 
const int Ngen_per_block = 5000; // No. of generations per block
const int Nsample = 100;         // No. of blocks to sample


// Explicitly typed constants so can easily work with both floats and floats
static const FLOAT DELTA = 2.0;         // Random step size
static const FLOAT FOUR = 4.0; 
static const FLOAT TWO  = 2.0;
static const FLOAT ONE  = 1.0;
static const FLOAT HALF = 0.5;
static const FLOAT ZERO = 0.0;

FLOAT rand (oneapi::mkl::rng::device::philox4x32x10<1> *state)
{
  oneapi::mkl::rng::device::uniform<FLOAT> distr_ct1;
  return oneapi::mkl::rng::device::generate(distr_ct1, *state);
}

FLOAT EXP(FLOAT x) {return cl::sycl::exp(x);}

FLOAT SQRT(FLOAT x) {return cl::sycl::sqrt(x);}


void SumWithinBlocks(const int n, global_ptr<FLOAT> data, 
                     global_ptr<FLOAT> blocksums, local_ptr<FLOAT> sdata, nd_item<1> &item) {
  int blockDim = item.get_local_range(0);
  int tid = item.get_local_id(0);
  int gid = item.get_group(0);
  int nthread =  blockDim * item.get_group_range(0); // blockDim.x*gridDim.x;
  int i = gid * blockDim + tid;  // global id

  // Every thread in every block computes partial sum over rest of vector
  FLOAT st = ZERO;
  while (i < n) {
    st += data[i];
    i += nthread;
  }
  sdata[tid] = st;
  item.barrier(access::fence_space::local_space);

  // Now do binary tree sum within a block
  
  // Round up to closest power of 2
  int pow2 = 1 << (32 - cl::sycl::clz(blockDim-1));
  
  for (unsigned int s=pow2>>1; s>0; s>>=1) {
    if (tid<s && (tid+s)<blockDim) {
      //printf("%4d : %4d %4d\n", tid, s, tid+s);
      sdata[tid] += sdata[tid + s];
    }
    item.barrier(access::fence_space::local_space);
  }
  if (tid==0) blocksums[gid] = sdata[0];
}

void compute_distances(const FLOAT x1, const FLOAT y1, const FLOAT z1, 
                       const FLOAT x2, const FLOAT y2, const FLOAT z2,
		       FLOAT& r1, FLOAT& r2, FLOAT& r12) {
    r1 = SQRT(x1*x1 + y1*y1 + z1*z1);
    r2 = SQRT(x2*x2 + y2*y2 + z2*z2);
    FLOAT xx = x1-x2;
    FLOAT yy = y1-y2;
    FLOAT zz = z1-z2;
    r12 = SQRT(xx*xx + yy*yy + zz*zz);
}

FLOAT psi(const FLOAT x1, const FLOAT y1, const FLOAT z1, const FLOAT x2, const FLOAT y2, const FLOAT z2) {
    FLOAT r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);
    return (ONE + HALF*r12)*EXP(-TWO*(r1 + r2));
}

// reset stats counters on the GPU
void zero_stats(global_ptr<FLOAT> stats, nd_item<1> &item) {
  int i = item.get_global_id(0);
  stats[0*Npoint+i] = ZERO; // r1
  stats[1*Npoint+i] = ZERO; // r2
  stats[2*Npoint+i] = ZERO; // r12
  stats[3*Npoint+i] = ZERO; // accept count
}

void propagate(const int nstep, global_ptr<FLOAT> X1, global_ptr<FLOAT> Y1, 
               global_ptr<FLOAT> Z1, global_ptr<FLOAT> X2, global_ptr<FLOAT> Y2, 
               global_ptr<FLOAT> Z2, global_ptr<FLOAT> P, global_ptr<FLOAT> stats, 
               global_ptr<oneapi::mkl::rng::device::philox4x32x10<1>> states, nd_item<1> &item) {
  int i = item.get_global_id(0);
  FLOAT x1 = X1[i];
  FLOAT y1 = Y1[i];
  FLOAT z1 = Z1[i];
  FLOAT x2 = X2[i];
  FLOAT y2 = Y2[i];
  FLOAT z2 = Z2[i];
  FLOAT p = P[i];
  
  for (int step=0; step<nstep; step++) {
    FLOAT x1new = x1 + (rand(states+i)-HALF)*DELTA;
    FLOAT y1new = y1 + (rand(states+i)-HALF)*DELTA;
    FLOAT z1new = z1 + (rand(states+i)-HALF)*DELTA;
    FLOAT x2new = x2 + (rand(states+i)-HALF)*DELTA;
    FLOAT y2new = y2 + (rand(states+i)-HALF)*DELTA;
    FLOAT z2new = z2 + (rand(states+i)-HALF)*DELTA;
    FLOAT pnew = psi(x1new, y1new, z1new, x2new, y2new, z2new);

    if (pnew*pnew > p*p*rand(states+i)) {
    	stats[3*Npoint+i]++; //naccept ++;
    	p = pnew;
    	x1 = x1new;
    	y1 = y1new;
    	z1 = z1new;
    	x2 = x2new;
    	y2 = y2new;
    	z2 = z2new;
    }
    
    FLOAT r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);
    
    stats[0*Npoint+i] += r1;
    stats[1*Npoint+i] += r2;
    stats[2*Npoint+i] += r12;
  }
  X1[i] = x1;  
  Y1[i] = y1;  
  Z1[i] = z1;  
  X2[i] = x2;  
  Y2[i] = y2;  
  Z2[i] = z2;  
  P[i] = p;
}
  
int main() {

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  
  buffer<FLOAT, 1> d_x1(Npoint);
  buffer<FLOAT, 1> d_y1(Npoint);
  buffer<FLOAT, 1> d_z1(Npoint);
  buffer<FLOAT, 1> d_x2(Npoint);
  buffer<FLOAT, 1> d_y2(Npoint);
  buffer<FLOAT, 1> d_z2(Npoint);
  buffer<FLOAT, 1> d_psir(Npoint);
  buffer<FLOAT, 1> d_stats(4*Npoint);
  buffer<FLOAT, 1> d_statsum(4);
  buffer<FLOAT, 1> d_blocksums(NBLOCK);
  buffer<oneapi::mkl::rng::device::philox4x32x10<1>, 1> d_ranstates(Npoint);

  //initran<<<NBLOCK,NTHR_PER_BLK>>>(5551212, ranstates);
  range<1> gws (Npoint);
  range<1> lws (NTHR_PER_BLK);

  q.submit([&](handler &cgh) {
    auto states = d_ranstates.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class init_random_states>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      int i = item.get_global_id(0);
      states[i] = oneapi::mkl::rng::device::philox4x32x10<1>(5551212, {0, static_cast<std::uint64_t>(i * 8)}); // ?? correct ??
    });
  });

  // initialize<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, x1, y1, z1, x2, y2, z2, psi, ranstates);
  q.submit([&](handler &cgh) {
    auto states = d_ranstates.get_access<sycl_read>(cgh);
    auto x1 = d_x1.get_access<sycl_discard_read_write>(cgh);
    auto y1 = d_y1.get_access<sycl_discard_read_write>(cgh);
    auto z1 = d_z1.get_access<sycl_discard_read_write>(cgh);
    auto x2 = d_x2.get_access<sycl_discard_read_write>(cgh);
    auto y2 = d_y2.get_access<sycl_discard_read_write>(cgh);
    auto z2 = d_z2.get_access<sycl_discard_read_write>(cgh);
    auto psir = d_psir.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class initialize>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      int i = item.get_global_id(0);
      x1[i] = (rand(states.get_pointer()+i) - HALF)*FOUR;
      y1[i] = (rand(states.get_pointer()+i) - HALF)*FOUR;
      z1[i] = (rand(states.get_pointer()+i) - HALF)*FOUR;
      x2[i] = (rand(states.get_pointer()+i) - HALF)*FOUR;
      y2[i] = (rand(states.get_pointer()+i) - HALF)*FOUR;
      z2[i] = (rand(states.get_pointer()+i) - HALF)*FOUR;
      psir[i] = psi(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]);
    });
  });

  //zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
  q.submit([&](handler &cgh) {
    auto stats = d_stats.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class reset_stats>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      zero_stats(stats.get_pointer(), item);
    });
  });
    
  // Equilibrate
  //propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Neq, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
    q.submit([&](handler &cgh) {
      auto x1 = d_x1.get_access<sycl_read_write>(cgh);
      auto y1 = d_y1.get_access<sycl_read_write>(cgh);
      auto z1 = d_z1.get_access<sycl_read_write>(cgh);
      auto x2 = d_x2.get_access<sycl_read_write>(cgh);
      auto y2 = d_y2.get_access<sycl_read_write>(cgh);
      auto z2 = d_z2.get_access<sycl_read_write>(cgh);
      auto psir = d_psir.get_access<sycl_read_write>(cgh);
      auto stats = d_stats.get_access<sycl_read_write>(cgh);
      auto ranstates = d_ranstates.get_access<sycl_read>(cgh);
      cgh.parallel_for<class prop2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        propagate(Neq, x1.get_pointer(), y1.get_pointer(), z1.get_pointer(),
             x2.get_pointer(), y2.get_pointer(), z2.get_pointer(),
             psir.get_pointer(), stats.get_pointer(), ranstates.get_pointer(), item);
      });
    });

  // Accumulators for averages over blocks --- use doubles
  double r1_tot = ZERO,  r1_sq_tot = ZERO;
  double r2_tot = ZERO,  r2_sq_tot = ZERO;
  double r12_tot = ZERO, r12_sq_tot = ZERO;
  double naccept = ZERO;  // Keeps track of propagation efficiency
  for (int sample=0; sample<Nsample; sample++) {
  //  zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
    q.submit([&](handler &cgh) {
      auto stats = d_stats.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class initialize>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        zero_stats(stats.get_pointer(), item);
      });
    });

    //propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Ngen_per_block, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
    q.submit([&](handler &cgh) {
      auto x1 = d_x1.get_access<sycl_read_write>(cgh);
      auto y1 = d_y1.get_access<sycl_read_write>(cgh);
      auto z1 = d_z1.get_access<sycl_read_write>(cgh);
      auto x2 = d_x2.get_access<sycl_read_write>(cgh);
      auto y2 = d_y2.get_access<sycl_read_write>(cgh);
      auto z2 = d_z2.get_access<sycl_read_write>(cgh);
      auto psir = d_psir.get_access<sycl_read_write>(cgh);
      auto stats = d_stats.get_access<sycl_read_write>(cgh);
      auto ranstates = d_ranstates.get_access<sycl_read>(cgh);
      cgh.parallel_for<class prop>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        propagate(Ngen_per_block, x1.get_pointer(), y1.get_pointer(), z1.get_pointer(),
             x2.get_pointer(), y2.get_pointer(), z2.get_pointer(),
             psir.get_pointer(), stats.get_pointer(), ranstates.get_pointer(), item);
      });
    });
    
    struct {FLOAT r1, r2, r12, accept;} s;
    //sum_stats(Npoint, stats, statsum, blocksums);
    for (int what=0; what<4; what++) {
      // SumWithinBlocks<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats+what*Npoint, blocksums);
      q.submit([&] (handler &cgh) {
        auto stats = d_stats.get_access<sycl_read>(cgh);
        auto blocksums = d_blocksums.get_access<sycl_write>(cgh);
        accessor<FLOAT, 1, sycl_read_write, access::target::local> sdata(512, cgh);
        cgh.parallel_for<class sum_blocks>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          SumWithinBlocks(Npoint, stats.get_pointer() + what * Npoint, 
                          blocksums.get_pointer(), sdata.get_pointer(), item); 
        });
      });

     //SumWithinBlocks<<<1,NBLOCK>>>(NBLOCK, blocksums, statsum+what);
      q.submit([&] (handler &cgh) {
        auto blocksums = d_blocksums.get_access<sycl_read>(cgh);
        auto statsum = d_statsum.get_access<sycl_write>(cgh);
        accessor<FLOAT, 1, sycl_read_write, access::target::local> sdata(512, cgh);
        cgh.parallel_for<class final_sum_blocks>(nd_range<1>(lws, lws), [=] (nd_item<1> item) {
          SumWithinBlocks(NBLOCK, blocksums.get_pointer(), 
                          statsum.get_pointer() + what, sdata.get_pointer(), item);
        });
      });
    }
  
    q.submit([&] (handler &cgh) {
      auto a = d_statsum.get_access<sycl_read>(cgh);  
      cgh.copy(a, &s);
    });
    q.wait();

    naccept += s.accept;
    s.r1 /= Ngen_per_block*Npoint;  
    s.r2 /= Ngen_per_block*Npoint;  
    s.r12 /= Ngen_per_block*Npoint;

    printf(" block %6d  %.6f  %.6f  %.6f\n", sample, s.r1, s.r2, s.r12);

    r1_tot += s.r1;   r1_sq_tot += s.r1*s.r1;
    r2_tot += s.r2;   r2_sq_tot += s.r2*s.r2;
    r12_tot += s.r12; r12_sq_tot += s.r12*s.r12;
  }

  r1_tot /= Nsample; r1_sq_tot /= Nsample; 
  r2_tot /= Nsample; r2_sq_tot /= Nsample; 
  r12_tot /= Nsample; r12_sq_tot /= Nsample; 
  
  double r1s = sqrt((r1_sq_tot - r1_tot*r1_tot) / Nsample);
  double r2s = sqrt((r2_sq_tot - r2_tot*r2_tot) / Nsample);
  double r12s = sqrt((r12_sq_tot - r12_tot*r12_tot) / Nsample);
  
  printf(" <r1>  = %.6f +- %.6f\n", r1_tot, r1s);
  printf(" <r2>  = %.6f +- %.6f\n", r2_tot, r2s);
  printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s);
  
  printf(" acceptance ratio=%.1f%%\n",100.0*naccept/double(Npoint)/double(Ngen_per_block)/double(Nsample)); // avoid int overflow
  return 0;
}
