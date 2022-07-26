#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <omp.h>

#define FLOAT float

// Number of threads per block
#define NTHR_PER_BLK 256
// Number of blocks
#define NBLOCK 56*4
// No. of independent samples
#define Npoint NBLOCK*NTHR_PER_BLK
// No. of generations to equilibrate 
#define Neq 100000
// No. of generations per block
#define Ngen_per_block 5000

#pragma omp declare target

// Explicitly typed constants so can easily work with both floats and floats
// Random step size
#define DELTA 2.f
#define FOUR  4.f
#define TWO   2.f
#define ONE   1.f
#define HALF  0.5f
#define ZERO  0.f

inline float EXP(float x) {return expf(x);}
inline double EXP(double x) {return exp(x);}
inline float SQRT(float x) {return sqrtf(x);}
inline double SQRT(double x) {return sqrt(x);}

float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
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
#pragma omp end declare target


void propagate(const int npoint, const int nstep, FLOAT* X1, FLOAT* Y1, FLOAT* Z1, 
               FLOAT* X2, FLOAT* Y2, FLOAT* Z2, FLOAT* P, FLOAT* stats, unsigned int* states)
{
  #pragma omp target teams distribute parallel for thread_limit(NTHR_PER_BLK)
  for (int i = 0; i < npoint; i++) {
    FLOAT x1 = X1[i];
    FLOAT y1 = Y1[i];
    FLOAT z1 = Z1[i];
    FLOAT x2 = X2[i];
    FLOAT y2 = Y2[i];
    FLOAT z2 = Z2[i];
    FLOAT p = P[i];
    
    for (int step=0; step<nstep; step++) {
      FLOAT x1new = x1 + (LCG_random(states+i)-HALF)*DELTA;
      FLOAT y1new = y1 + (LCG_random(states+i)-HALF)*DELTA;
      FLOAT z1new = z1 + (LCG_random(states+i)-HALF)*DELTA;
      FLOAT x2new = x2 + (LCG_random(states+i)-HALF)*DELTA;
      FLOAT y2new = y2 + (LCG_random(states+i)-HALF)*DELTA;
      FLOAT z2new = z2 + (LCG_random(states+i)-HALF)*DELTA;
      FLOAT pnew = wave_function(x1new, y1new, z1new, x2new, y2new, z2new);

      if (pnew*pnew > p*p*LCG_random(states+i)) {
        stats[3*npoint+i]++; //naccept ++;
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
      
      stats[0*npoint+i] += r1;
      stats[1*npoint+i] += r2;
      stats[2*npoint+i] += r12;
    }
    X1[i] = x1;  
    Y1[i] = y1;  
    Z1[i] = z1;  
    X2[i] = x2;  
    Y2[i] = y2;  
    Z2[i] = z2;  
    P[i] = p;
  }
}

// Initialize random number generator
void initran(const int npoint, unsigned int seed, unsigned int* states) {
  #pragma omp target teams distribute parallel for thread_limit(NTHR_PER_BLK)
  for (int i = 0; i < npoint; i++) {
    states[i] = seed ^ i;
    LCG_random_init(&states[i]);
  }
}

void initialize(const int npoint, FLOAT* x1, FLOAT* y1, FLOAT* z1, 
                FLOAT* x2, FLOAT* y2, FLOAT* z2, FLOAT* psi, unsigned int* states) {
  #pragma omp target teams distribute parallel for thread_limit(NTHR_PER_BLK)
  for (int i = 0; i < npoint; i++) {
    x1[i] = (LCG_random(states+i) - HALF)*FOUR;
    y1[i] = (LCG_random(states+i) - HALF)*FOUR;
    z1[i] = (LCG_random(states+i) - HALF)*FOUR;
    x2[i] = (LCG_random(states+i) - HALF)*FOUR;
    y2[i] = (LCG_random(states+i) - HALF)*FOUR;
    z2[i] = (LCG_random(states+i) - HALF)*FOUR;
    psi[i] = wave_function(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]);
  }
}

// ZERO stats counters on the GPU
void zero_stats(const int npoint, FLOAT* stats) {
  #pragma omp target teams distribute parallel for thread_limit(NTHR_PER_BLK)
  for (int i = 0; i < npoint; i++) {
    stats[0*npoint+i] = ZERO; // r1
    stats[1*npoint+i] = ZERO; // r2
    stats[2*npoint+i] = ZERO; // r12
    stats[3*npoint+i] = ZERO; // accept count
  }
}

void SumWithinBlocks(const int n, const int threads, FLOAT *data, FLOAT* blocksums)
{
  const int teams = n / threads;
  #pragma omp target teams num_teams(teams) thread_limit(threads)
  { 
     FLOAT sdata[512];
     #pragma omp parallel 
     {
       int blockDim = omp_get_num_threads();
       int tid = omp_get_thread_num();
       int gid = omp_get_team_num();
       int nthread =  blockDim * omp_get_num_teams(); // blockDim.x*gridDim.x;
       int i = gid * blockDim + tid;  // global id

       // Every thread in every block computes partial sum over rest of vector
       FLOAT st = ZERO;
       while (i < n) {
         st += data[i];
         i += nthread;
       }
       sdata[tid] = st;
       #pragma omp barrier

       // Now do binary tree sum within a block
       
       for (unsigned int s=128; s>0; s>>=1) {
         if (tid<s && (tid+s)<blockDim) {
           sdata[tid] += sdata[tid + s];
         }
         #pragma omp barrier
       }
       if (tid==0) blocksums[gid] = sdata[0];
     }
   }
}
  
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <number of blocks to sample>\n", argv[0]);
    return 1;
  }
  const int Nsample = atoi(argv[1]); // No. of blocks to sample
  
  FLOAT *x1 = (FLOAT*) malloc(Npoint * sizeof(FLOAT));
  FLOAT *y1 = (FLOAT*) malloc(Npoint * sizeof(FLOAT));
  FLOAT *z1 = (FLOAT*) malloc(Npoint * sizeof(FLOAT));
  FLOAT *x2 = (FLOAT*) malloc(Npoint * sizeof(FLOAT));
  FLOAT *y2 = (FLOAT*) malloc(Npoint * sizeof(FLOAT));
  FLOAT *z2 = (FLOAT*) malloc(Npoint * sizeof(FLOAT));
  FLOAT *psi = (FLOAT*) malloc(Npoint * sizeof(FLOAT));
  FLOAT *stats = (FLOAT*) malloc(4 * Npoint * sizeof(FLOAT));
  FLOAT *statsum = (FLOAT*) malloc(4 * sizeof(FLOAT));
  FLOAT *blocksums = (FLOAT*) malloc(NBLOCK * sizeof(FLOAT));
  unsigned int *ranstates = (unsigned int*) malloc(Npoint * sizeof(unsigned int));

  #pragma omp target data map(alloc:x1[0:Npoint],\
                                    y1[0:Npoint],\
                                    z1[0:Npoint],\
                                    x2[0:Npoint],\
                                    y2[0:Npoint],\
                                    z2[0:Npoint],\
                                    psi[0:Npoint],\
                                    stats[0:4*Npoint],\
                                    statsum[0:4],\
                                    blocksums[0:NBLOCK],\
                                    ranstates[0:Npoint])
  {
    initran(Npoint, 5551212, ranstates);
  
    initialize(Npoint, x1, y1, z1, x2, y2, z2, psi, ranstates);
  
    zero_stats(Npoint, stats);
      
    // Equilibrate
    propagate(Npoint, Neq, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
  
    // Accumulators for averages over blocks --- use doubles
    double r1_tot = ZERO,  r1_sq_tot = ZERO;
    double r2_tot = ZERO,  r2_sq_tot = ZERO;
    double r12_tot = ZERO, r12_sq_tot = ZERO;
    double naccept = ZERO;  // Keeps track of propagation efficiency
  
    double time = 0.0;
  
    for (int sample=0; sample<Nsample; sample++) {
      auto start = std::chrono::steady_clock::now();
  
      zero_stats(Npoint, stats);
  
      propagate(Npoint, Ngen_per_block, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
  
      //sum_stats(Npoint, stats, statsum, blocksums);
      for (int what=0; what<4; what++) {
        SumWithinBlocks(Npoint, NTHR_PER_BLK, stats+what*Npoint, blocksums);
        SumWithinBlocks(NBLOCK, NBLOCK, blocksums, statsum+what);
      }
  
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
      #pragma omp target update from (statsum[0:4])
      struct {FLOAT r1, r2, r12, accept;} s;
      memcpy(&s, statsum, 4 * sizeof(FLOAT));
  
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
  }

  free(x1);
  free(y1);
  free(z1);
  free(x2);
  free(y2);
  free(z2);
  free(psi);
  free(stats);
  free(blocksums);
  free(statsum);
  free(ranstates);
  return 0;
}
