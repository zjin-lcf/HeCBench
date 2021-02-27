#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cmath>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

using namespace std;

using FLOAT = float;

const int NTHR_PER_BLK = 256; // Number of CUDA threads per block
const int NBLOCK  = 56*4;  // Number of CUDA blocks (SMs on P100)

#define CHECK(test) if (test != 0) throw "error";

const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Neq = 100000;          // No. of generations to equilibrate 

const int Ngen_per_block = 5000; // No. of generations per block
const int Nsample = 100;         // No. of blocks to sample

const float DELTA = 2.0;        // Random step size

// Explicitly typed constants so can easily work with both floats and floats
static const FLOAT FOUR = 4.0; 
static const FLOAT TWO  = 2.0;
static const FLOAT ONE  = 1.0;
static const FLOAT HALF = 0.5;
static const FLOAT ZERO = 0.0;


/*
DPCT1032:1: Different generator is used, you may need to adjust the code.
*/
template <typename T>
__dpct_inline__ T rand(oneapi::mkl::rng::device::philox4x32x10<1> *state) {
  oneapi::mkl::rng::device::uniform<T> distr_ct1;
  return oneapi::mkl::rng::device::generate(distr_ct1, *state);
}


__dpct_inline__ float EXP(float x) { return sycl::exp(x); }
__dpct_inline__ double EXP(double x) { return sycl::exp(x); }
__dpct_inline__ float SQRT(float x) { return sycl::sqrt(x); }
__dpct_inline__ double SQRT(double x) { return sycl::sqrt(x); }

void SumWithinBlocks(const int n, const FLOAT* data, FLOAT* blocksums,
                     sycl::nd_item<3> item_ct1, FLOAT *sdata) {
  int nthread = item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);
  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
    // max threads

  // Every thread in every block computes partial sum over rest of vector
  FLOAT st=ZERO;
  while (i < n) {
    st += data[i];
    i+=nthread;
  }
  sdata[item_ct1.get_local_id(2)] = st;
  item_ct1.barrier();

  // Now do binary tree sum within a block

  // Round up to closest power of 2
  int pow2 = 1 << (32 - sycl::clz((int)(item_ct1.get_local_range(2) - 1)));

  int tid = item_ct1.get_local_id(2);
  for (unsigned int s=pow2>>1; s>0; s>>=1) {
    if (tid < s && (tid + s) < item_ct1.get_local_range().get(2)) {
      //printf("%4d : %4d %4d\n", tid, s, tid+s);
      sdata[tid] += sdata[tid + s];
    }
    item_ct1.barrier();
  }
  if (tid == 0) blocksums[item_ct1.get_group(2)] = sdata[0];
}

void sum_stats(const int Npoint, const FLOAT *stats, FLOAT *statsum,
               FLOAT *blocksums) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  for (int what=0; what<4; what++) {
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<FLOAT, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          sdata_acc_ct1(sycl::range<1>(512), cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK) *
                                             sycl::range<3>(1, 1, NTHR_PER_BLK),
                                         sycl::range<3>(1, 1, NTHR_PER_BLK)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SumWithinBlocks(Npoint, stats + what * Npoint,
                                         blocksums, item_ct1,
                                         sdata_acc_ct1.get_pointer());
                       });
    });
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<FLOAT, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          sdata_acc_ct1(sycl::range<1>(512), cgh);

      auto NBLOCK_ct0 = NBLOCK;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK),
                                         sycl::range<3>(1, 1, NBLOCK)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SumWithinBlocks(NBLOCK_ct0, blocksums, statsum + what,
                                         item_ct1, sdata_acc_ct1.get_pointer());
                       });
    });
  }
}

__dpct_inline__ void compute_distances(FLOAT x1, FLOAT y1, FLOAT z1, FLOAT x2,
                                       FLOAT y2, FLOAT z2, FLOAT &r1, FLOAT &r2,
                                       FLOAT &r12) {
  r1 = SQRT(x1*x1 + y1*y1 + z1*z1);
  r2 = SQRT(x2*x2 + y2*y2 + z2*z2);
  FLOAT xx = x1-x2;
  FLOAT yy = y1-y2;
  FLOAT zz = z1-z2;
  r12 = SQRT(xx*xx + yy*yy + zz*zz);
}

__dpct_inline__ FLOAT psi(FLOAT x1, FLOAT y1, FLOAT z1, FLOAT x2, FLOAT y2,
                          FLOAT z2) {
  FLOAT r1, r2, r12;
  compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);

  return (ONE + HALF*r12)*EXP(-TWO*(r1 + r2));
}

// Initialize random number generator
/*
DPCT1032:4: Different generator is used, you may need to adjust the code.
*/
void initran(unsigned int seed,
             oneapi::mkl::rng::device::philox4x32x10<1> *states,
             sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
  states[i] = oneapi::mkl::rng::device::philox4x32x10<1>(
      seed, {0, static_cast<std::uint64_t>(i * 8)});
}

// ZERO stats counters on the GPU
void zero_stats(int Npoint, FLOAT* stats, sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
  stats[0*Npoint+i] = ZERO; // r1
  stats[1*Npoint+i] = ZERO; // r2
  stats[2*Npoint+i] = ZERO; // r12
  stats[3*Npoint+i] = ZERO; // accept count
}

// initializes samples
/*
DPCT1032:5: Different generator is used, you may need to adjust the code.
*/
void initialize(FLOAT *x1, FLOAT *y1, FLOAT *z1, FLOAT *x2, FLOAT *y2,
                FLOAT *z2, FLOAT *psir,
                oneapi::mkl::rng::device::philox4x32x10<1> *states,
                sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
  x1[i] = (rand<FLOAT>(states+i) - HALF)*FOUR;
  y1[i] = (rand<FLOAT>(states+i) - HALF)*FOUR;
  z1[i] = (rand<FLOAT>(states+i) - HALF)*FOUR;
  x2[i] = (rand<FLOAT>(states+i) - HALF)*FOUR;
  y2[i] = (rand<FLOAT>(states+i) - HALF)*FOUR;
  z2[i] = (rand<FLOAT>(states+i) - HALF)*FOUR;
  psir[i] = psi(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]);
}

void propagate(
    const int Npoint, const int nstep, FLOAT *X1, FLOAT *Y1, FLOAT *Z1,
    /*
    DPCT1032:6: Different generator is used, you may need to adjust the code.
    */
    FLOAT *X2, FLOAT *Y2, FLOAT *Z2, FLOAT *P, FLOAT *stats,
    oneapi::mkl::rng::device::philox4x32x10<1> *states,
    sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
  FLOAT x1 = X1[i];
  FLOAT y1 = Y1[i];
  FLOAT z1 = Z1[i];
  FLOAT x2 = X2[i];
  FLOAT y2 = Y2[i];
  FLOAT z2 = Z2[i];
  FLOAT p = P[i];

  for (int step=0; step<nstep; step++) {
    FLOAT x1new = x1 + (rand<FLOAT>(states+i)-HALF)*DELTA;
    FLOAT y1new = y1 + (rand<FLOAT>(states+i)-HALF)*DELTA;
    FLOAT z1new = z1 + (rand<FLOAT>(states+i)-HALF)*DELTA;
    FLOAT x2new = x2 + (rand<FLOAT>(states+i)-HALF)*DELTA;
    FLOAT y2new = y2 + (rand<FLOAT>(states+i)-HALF)*DELTA;
    FLOAT z2new = z2 + (rand<FLOAT>(states+i)-HALF)*DELTA;
    FLOAT pnew = psi(x1new, y1new, z1new, x2new, y2new, z2new);

    if (pnew*pnew > p*p*rand<FLOAT>(states+i)) {
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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  FLOAT *x1, *y1, *z1, *x2, *y2, *z2, *psi, *stats, *statsum, *blocksums;
  /*
  DPCT1032:7: Different generator is used, you may need to adjust the code.
  */
  oneapi::mkl::rng::device::philox4x32x10<1> *ranstates;

  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((x1 = sycl::malloc_device<FLOAT>(Npoint, q_ct1), 0));
  /*
  DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((y1 = sycl::malloc_device<FLOAT>(Npoint, q_ct1), 0));
  /*
  DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((z1 = sycl::malloc_device<FLOAT>(Npoint, q_ct1), 0));
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((x2 = sycl::malloc_device<FLOAT>(Npoint, q_ct1), 0));
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((y2 = sycl::malloc_device<FLOAT>(Npoint, q_ct1), 0));
  /*
  DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((z2 = sycl::malloc_device<FLOAT>(Npoint, q_ct1), 0));
  /*
  DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((psi = sycl::malloc_device<FLOAT>(Npoint, q_ct1), 0));
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((stats = sycl::malloc_device<FLOAT>(4 * Npoint, q_ct1), 0));
  /*
  DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((blocksums = sycl::malloc_device<FLOAT>(NBLOCK, q_ct1),
         0)); // workspace for summation
  /*
  DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((statsum = sycl::malloc_device<FLOAT>(4, q_ct1),
         0)); // workspace for summation
  /*
  DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  /*
  DPCT1032:19: Different generator is used, you may need to adjust the code.
  */
  CHECK((ranstates =
             sycl::malloc_device<oneapi::mkl::rng::device::philox4x32x10<1>>(
                 Npoint, q_ct1),
         0));

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK) *
                                           sycl::range<3>(1, 1, NTHR_PER_BLK),
                                       sycl::range<3>(1, 1, NTHR_PER_BLK)),
                     [=](sycl::nd_item<3> item_ct1) {
                       initran(5551212, ranstates, item_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK) *
                                           sycl::range<3>(1, 1, NTHR_PER_BLK),
                                       sycl::range<3>(1, 1, NTHR_PER_BLK)),
                     [=](sycl::nd_item<3> item_ct1) {
                       initialize(x1, y1, z1, x2, y2, z2, psi, ranstates,
                                  item_ct1);
                     });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
    auto Npoint_ct0 = Npoint;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK) *
                                           sycl::range<3>(1, 1, NTHR_PER_BLK),
                                       sycl::range<3>(1, 1, NTHR_PER_BLK)),
                     [=](sycl::nd_item<3> item_ct1) {
                       zero_stats(Npoint_ct0, stats, item_ct1);
                     });
  });

  // Equilibrate
  q_ct1.submit([&](sycl::handler &cgh) {
    auto Npoint_ct0 = Npoint;
    auto Neq_ct1 = Neq;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK) *
                                           sycl::range<3>(1, 1, NTHR_PER_BLK),
                                       sycl::range<3>(1, 1, NTHR_PER_BLK)),
                     [=](sycl::nd_item<3> item_ct1) {
                       propagate(Npoint_ct0, Neq_ct1, x1, y1, z1, x2, y2, z2,
                                 psi, stats, ranstates, item_ct1);
                     });
  });

  // Accumulators for averages over blocks --- use doubles
  double r1_tot = ZERO,  r1_sq_tot = ZERO;
  double r2_tot = ZERO,  r2_sq_tot = ZERO;
  double r12_tot = ZERO, r12_sq_tot = ZERO;
  double naccept = ZERO;  // Keeps track of propagation efficiency
  for (int sample=0; sample<Nsample; sample++) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto Npoint_ct0 = Npoint;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK) *
                                             sycl::range<3>(1, 1, NTHR_PER_BLK),
                                         sycl::range<3>(1, 1, NTHR_PER_BLK)),
                       [=](sycl::nd_item<3> item_ct1) {
                         zero_stats(Npoint_ct0, stats, item_ct1);
                       });
    });
    q_ct1.submit([&](sycl::handler &cgh) {
      auto Npoint_ct0 = Npoint;
      auto Ngen_per_block_ct1 = Ngen_per_block;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCK) *
                                             sycl::range<3>(1, 1, NTHR_PER_BLK),
                                         sycl::range<3>(1, 1, NTHR_PER_BLK)),
                       [=](sycl::nd_item<3> item_ct1) {
                         propagate(Npoint_ct0, Ngen_per_block_ct1, x1, y1, z1,
                                   x2, y2, z2, psi, stats, ranstates, item_ct1);
                       });
    });

    struct {FLOAT r1, r2, r12, accept;} s;
    sum_stats(Npoint, stats, statsum, blocksums);
    /*
    DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CHECK((q_ct1.memcpy(&s, statsum, sizeof(s)).wait(), 0));

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

  /*
  DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(x1, q_ct1), 0));
  /*
  DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(y1, q_ct1), 0));
  /*
  DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(z1, q_ct1), 0));
  /*
  DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(x2, q_ct1), 0));
  /*
  DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(y2, q_ct1), 0));
  /*
  DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(z2, q_ct1), 0));
  /*
  DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(psi, q_ct1), 0));
  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(stats, q_ct1), 0));
  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(blocksums, q_ct1), 0));
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(statsum, q_ct1), 0));
  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CHECK((sycl::free(ranstates, q_ct1), 0));
  return 0;
}

