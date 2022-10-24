#include <chrono>
#include <omp.h>
#include "kernel.h"

float fitness_function(float x[])
{
  float res = 0.f;
  float y1 = F(x[0]);
  float yn = F(x[DIM-1]);

  res += powf(sinf(phi*y1), 2.f) + powf(yn-1, 2.f);

  for(int i = 0; i < DIM-1; i++)
  {
    float y = F(x[i]);
    float yp = F(x[i+1]);
    res += powf(y-1.f, 2.f) * (1.f + 10.f * powf(sinf(phi*yp), 2.f));
  }

  return res;
}

void kernelUpdateParticle(float *__restrict positions,
                          float *__restrict velocities,
                          const float *__restrict pBests,
                          const float *__restrict gBest,
                          const int p,
                          const float rp,
                          const float rg)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i=0; i < p*DIM; i++) {
    velocities[i]=OMEGA*velocities[i]+
                  c1*rp*(pBests[i]-positions[i])+
                  c2*rg*(gBest[i%DIM]-positions[i]);
    positions[i]+=velocities[i];
  }
}

void kernelUpdatePBest(const float *__restrict positions,
                             float *__restrict pBests,
                             float *__restrict gBest,
                       const int p)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i=0; i < p; i++) {
    i = i*DIM;

    float tempParticle1[DIM];
    float tempParticle2[DIM];

    for(int j=0;j<DIM;j++)
    {
      tempParticle1[j]=positions[i+j];
      tempParticle2[j]=pBests[i+j];
    }

    if(fitness_function(tempParticle1)<fitness_function(tempParticle2))
    {
      for(int j=0;j<DIM;j++)
        pBests[i+j]=tempParticle1[j];

      if(fitness_function(tempParticle1)<130.f) //fitness_function(gBest))
      {
        for(int j=0;j<DIM;j++) {
          #pragma omp atomic
          gBest[j] += tempParticle1[j];
        }
      }
    }
  }
}

extern "C" void gpu_pso(int p, int r,
                        float *positions,float *velocities,float *pBests,float *gBest)
{
  int size = p*DIM;

  #pragma omp target data map(to: positions[0:size],velocities[0:size]) \
                          map(tofrom: gBest[0:DIM], pBests[0:size])
  {
    auto start = std::chrono::steady_clock::now();

    for(int iter=0;iter<r;iter++)
    {
      float rp=getRandomClamped(iter);
      float rg=getRandomClamped(r-iter);
      kernelUpdateParticle(positions,
                           velocities,
                           pBests,
                           gBest,
                           p,rp,rg);

      kernelUpdatePBest(positions,pBests,gBest,p);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", time * 1e-3f / r);
  }
}
