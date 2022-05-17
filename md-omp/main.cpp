#include <cassert>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <list>
#include <iostream>
#include <omp.h>
#include "MD.h"
#include "reference.h"
#include "utils.h"

void md (
    const POSVECTYPE* __restrict position,
    FORCEVECTYPE* __restrict force,
    const int* __restrict neighborList, 
    const int nAtom,
    const int maxNeighbors, 
    const FPTYPE lj1_t,
    const FPTYPE lj2_t,
    const FPTYPE cutsq_t )
{
  #pragma omp target teams distribute parallel for thread_limit(256) 
  for (uint idx = 0; idx < nAtom; idx++) {
    POSVECTYPE ipos = position[idx];
    FORCEVECTYPE f = zero;

    int j = 0;
    while (j < maxNeighbors)
    {
      int jidx = neighborList[j*nAtom + idx];

      // Uncoalesced read
      POSVECTYPE jpos = position[jidx];

      // Calculate distance
      FPTYPE delx = ipos.x - jpos.x;
      FPTYPE dely = ipos.y - jpos.y;
      FPTYPE delz = ipos.z - jpos.z;
      FPTYPE r2inv = delx*delx + dely*dely + delz*delz;

      // If distance is less than cutoff, calculate force
      if (r2inv > 0 && r2inv < cutsq_t)
      {
        r2inv = (FPTYPE)1.0 / r2inv;
        FPTYPE r6inv = r2inv * r2inv * r2inv;
        FPTYPE forceC = r2inv*r6inv*(lj1_t*r6inv - lj2_t);

        f.x += delx * forceC;
        f.y += dely * forceC;
        f.z += delz * forceC;
      }
      j++;
    }
    force[idx] = f;
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("usage: %s <class size> <iteration>", argv[0]);
    return 1;
  }

  // Problem Parameters
  int sizeClass = atoi(argv[1]);
  int iteration = atoi(argv[2]);
  const int probSizes[] = { 12288, 24576, 36864, 73728 };
  assert(sizeClass >= 0 && sizeClass < 4);
  assert(iteration >= 0);

  int nAtom = probSizes[sizeClass];

  // Allocate problem data on host
  POSVECTYPE* position = (POSVECTYPE*) malloc(nAtom * sizeof(POSVECTYPE));
  FORCEVECTYPE* force = (FORCEVECTYPE*) malloc(nAtom * sizeof(FORCEVECTYPE));
  int *neighborList = (int*) malloc(maxNeighbors * nAtom * sizeof(int));

  std::cout << "Initializing test problem (this can take several minutes for large problems).\n";

  // Seed random number generator
  srand(123);

  // Notes on positions 
  // When the potential energy becomes exceedingly large as the distance 
  // between two atoms is very close, the host and device results may differ significantly
  for (int i = 0; i < nAtom; i++)
  {
    position[i].x = rand() % domainEdge;
    position[i].y = rand() % domainEdge;
    position[i].z = rand() % domainEdge;
  }

  std::cout << "Finished.\n";
  int totalPairs = buildNeighborList<FPTYPE, POSVECTYPE>(nAtom, position, neighborList);
  std::cout << totalPairs << " of " << nAtom*maxNeighbors
            << " pairs within cutoff distance = "
            << 100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %\n";

  #pragma omp target data map(to: position[0:nAtom], \
                                  neighborList[0:nAtom * maxNeighbors]) \
                          map(alloc: force[0:nAtom])
  {
    // warmup and result verification
    md(position,
       force,
       neighborList,
       nAtom,
       maxNeighbors,
       lj1,
       lj2,
       cutsq);

    #pragma omp target update from(force[0:nAtom])

    std::cout << "Performing Correctness Check (may take several minutes)\n";

    checkResults<FPTYPE, FORCEVECTYPE, POSVECTYPE>(force, position, neighborList, nAtom);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iteration; i++) {
      md(position,
         force,
         neighborList,
         nAtom,
         maxNeighbors,
         lj1,
         lj2,
         cutsq);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / iteration << " (s)\n";
  }

  free(position);
  free(force);
  free(neighborList);

  return 0;
}
