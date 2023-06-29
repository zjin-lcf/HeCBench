#include <chrono>
#include <cstdio>
#include <cmath>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <sycl/sycl.hpp>

using FLOAT = float;

const int NTHR_PER_BLK = 256;           // Number of threads per block
const int NBLOCK  = 56*4;               // Number of blocks
const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Neq = 100000;                 // No. of generations to equilibrate
const int Ngen_per_block = 5000;        // No. of generations per block

// Explicitly typed constants so can easily work with both floats and floats
const FLOAT DELTA = 2.0;         // Random step size
const FLOAT FOUR = 4.0;
const FLOAT TWO  = 2.0;
const FLOAT ONE  = 1.0;
const FLOAT HALF = 0.5;
const FLOAT ZERO = 0.0;

FLOAT rand (oneapi::mkl::rng::device::philox4x32x10<1> *state)
{
  oneapi::mkl::rng::device::uniform<FLOAT> distr_ct1;
  return oneapi::mkl::rng::device::generate(distr_ct1, *state);
}

FLOAT EXP(FLOAT x) {return sycl::exp(x);}

FLOAT SQRT(FLOAT x) {return sycl::sqrt(x);}

void SumWithinBlocks(const int n, FLOAT *data,
                     FLOAT *blocksums, FLOAT *sdata, sycl::nd_item<1> &item)
{
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
  item.barrier(sycl::access::fence_space::local_space);

  // Now do binary tree sum within a block

  // Round up to closest power of 2
  int pow2 = 1 << (32 - sycl::clz(blockDim-1));

  for (unsigned int s=pow2>>1; s>0; s>>=1) {
    if (tid<s && (tid+s)<blockDim) {
      //printf("%4d : %4d %4d\n", tid, s, tid+s);
      sdata[tid] += sdata[tid + s];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
  if (tid==0) blocksums[gid] = sdata[0];
}

void compute_distances(const FLOAT x1, const FLOAT y1, const FLOAT z1,
                       const FLOAT x2, const FLOAT y2, const FLOAT z2,
		       FLOAT& r1, FLOAT& r2, FLOAT& r12)
{
    r1 = SQRT(x1*x1 + y1*y1 + z1*z1);
    r2 = SQRT(x2*x2 + y2*y2 + z2*z2);
    FLOAT xx = x1-x2;
    FLOAT yy = y1-y2;
    FLOAT zz = z1-z2;
    r12 = SQRT(xx*xx + yy*yy + zz*zz);
}

FLOAT wave_function(const FLOAT x1, const FLOAT y1, const FLOAT z1, const FLOAT x2, const FLOAT y2, const FLOAT z2)
{
    FLOAT r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);
    return (ONE + HALF*r12)*EXP(-TWO*(r1 + r2));
}

// reset stats counters on the GPU
void zero_stats(FLOAT *stats, sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  stats[0*Npoint+i] = ZERO; // r1
  stats[1*Npoint+i] = ZERO; // r2
  stats[2*Npoint+i] = ZERO; // r12
  stats[3*Npoint+i] = ZERO; // accept count
}

void propagate(const int nstep, FLOAT *X1, FLOAT *Y1,
               FLOAT *Z1, FLOAT *X2, FLOAT *Y2,
               FLOAT *Z2, FLOAT *P, FLOAT *stats,
               oneapi::mkl::rng::device::philox4x32x10<1> *states,
               sycl::nd_item<1> &item)
{
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
    FLOAT pnew = wave_function(x1new, y1new, z1new, x2new, y2new, z2new);

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

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <number of blocks to sample>\n", argv[0]);
    return 1;
  }
  const int Nsample = atoi(argv[1]); // No. of blocks to sample

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  FLOAT *d_x1 = sycl::malloc_device<FLOAT>(Npoint, q);
  FLOAT *d_y1 = sycl::malloc_device<FLOAT>(Npoint, q);
  FLOAT *d_z1 = sycl::malloc_device<FLOAT>(Npoint, q);
  FLOAT *d_x2 = sycl::malloc_device<FLOAT>(Npoint, q);
  FLOAT *d_y2 = sycl::malloc_device<FLOAT>(Npoint, q);
  FLOAT *d_z2 = sycl::malloc_device<FLOAT>(Npoint, q);
  FLOAT *d_psi = sycl::malloc_device<FLOAT>(Npoint, q);
  FLOAT *d_stats = sycl::malloc_device<FLOAT>(4*Npoint, q);
  FLOAT *d_statsum = sycl::malloc_device<FLOAT>(4, q);
  FLOAT *d_blocksums = sycl::malloc_device<FLOAT>(NBLOCK, q);
  oneapi::mkl::rng::device::philox4x32x10<1> *d_ranstates =
    sycl::malloc_device<oneapi::mkl::rng::device::philox4x32x10<1>>(Npoint, q);

  //initran<<<NBLOCK,NTHR_PER_BLK>>>(5551212, ranstates);
  sycl::range<1> gws (Npoint);
  sycl::range<1> lws (NTHR_PER_BLK);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class init_random_states>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i = item.get_global_id(0);
      d_ranstates[i] = oneapi::mkl::rng::device::philox4x32x10<1>(
                       5551212, {0, static_cast<std::uint64_t>(i * 8)});
    });
  });

  // initialize<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, x1, y1, z1, x2, y2, z2, psi, ranstates);
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class initialize>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i = item.get_global_id(0);
      d_x1[i] = (rand(d_ranstates+i) - HALF)*FOUR;
      d_y1[i] = (rand(d_ranstates+i) - HALF)*FOUR;
      d_z1[i] = (rand(d_ranstates+i) - HALF)*FOUR;
      d_x2[i] = (rand(d_ranstates+i) - HALF)*FOUR;
      d_y2[i] = (rand(d_ranstates+i) - HALF)*FOUR;
      d_z2[i] = (rand(d_ranstates+i) - HALF)*FOUR;
      d_psi[i] = wave_function(d_x1[i], d_y1[i], d_z1[i], d_x2[i], d_y2[i], d_z2[i]);
    });
  });

  //zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class reset_stats>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      zero_stats(d_stats, item);
    });
  });

  // Equilibrate
  //propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Neq, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class prop2>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      propagate(Neq, d_x1, d_y1, d_z1,
                d_x2, d_y2, d_z2,
                d_psi, d_stats, d_ranstates, item);
    });
  });

  // Accumulators for averages over blocks --- use doubles
  double r1_tot = ZERO,  r1_sq_tot = ZERO;
  double r2_tot = ZERO,  r2_sq_tot = ZERO;
  double r12_tot = ZERO, r12_sq_tot = ZERO;
  double naccept = ZERO;  // Keeps track of propagation efficiency

  double time = 0.0;

  for (int sample=0; sample<Nsample; sample++) {
    q.wait();
    auto start = std::chrono::steady_clock::now();

    //  zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class zero>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        zero_stats(d_stats, item);
      });
    });

    //propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Ngen_per_block, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class prop>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        propagate(Ngen_per_block, d_x1, d_y1, d_z1,
                  d_x2, d_y2, d_z2,
                  d_psi, d_stats, d_ranstates, item);
      });
    });

    //sum_stats(Npoint, stats, statsum, blocksums);
    for (int what=0; what<4; what++) {
      // SumWithinBlocks<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats+what*Npoint, blocksums);
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<FLOAT, 1> sdata (sycl::range<1>(512), cgh);
        cgh.parallel_for<class sum_blocks>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          SumWithinBlocks(Npoint, d_stats + what * Npoint,
                          d_blocksums, sdata.get_pointer(), item);
        });
      });

     //SumWithinBlocks<<<1,NBLOCK>>>(NBLOCK, blocksums, statsum+what);
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<FLOAT, 1> sdata (sycl::range<1>(512), cgh);
        cgh.parallel_for<class final_sum_blocks>(
          sycl::nd_range<1>(lws, lws), [=] (sycl::nd_item<1> item) {
          SumWithinBlocks(NBLOCK, d_blocksums,
                          d_statsum + what, sdata.get_pointer(), item);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    struct {FLOAT r1, r2, r12, accept;} s;
    q.memcpy(&s, d_statsum, sizeof(s)).wait();

    naccept += s.accept;
    s.r1 /= Ngen_per_block*Npoint;
    s.r2 /= Ngen_per_block*Npoint;
    s.r12 /= Ngen_per_block*Npoint;

#ifdef DEBUG
    printf(" block %6d  %.6f  %.6f  %.6f\n", sample, s.r1, s.r2, s.r12);
#endif

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

  // avoid int overflow
  printf(" acceptance ratio=%.1f%%\n",
    100.0*naccept/double(Npoint)/double(Ngen_per_block)/double(Nsample));

  printf("Average execution time of kernels: %f (s)\n", (time * 1e-9f) / Nsample);

  sycl::free(d_x1, q);
  sycl::free(d_y1, q);
  sycl::free(d_z1, q);
  sycl::free(d_x2, q);
  sycl::free(d_y2, q);
  sycl::free(d_z2, q);
  sycl::free(d_psi, q);
  sycl::free(d_stats, q);
  sycl::free(d_blocksums, q);
  sycl::free(d_statsum, q);
  sycl::free(d_ranstates, q);
  return 0;
}
