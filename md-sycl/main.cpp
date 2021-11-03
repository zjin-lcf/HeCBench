#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <list>
#include <iostream>
#include "common.h"
#include "MD.h"
#include "reference.h"
#include "utils.h"

void md (
  nd_item<1> &item,
  const POSVECTYPE* __restrict position,
        FORCEVECTYPE* __restrict force,
  const int* __restrict neighborList, 
  const int nAtom,
  const int maxNeighbors, 
  const FPTYPE lj1_t,
  const FPTYPE lj2_t,
  const FPTYPE cutsq_t )
{
  const uint idx = item.get_global_id(0);
  if (idx >= nAtom) return;

  POSVECTYPE ipos = position[idx];
  FORCEVECTYPE f = FORCEVECTYPE(0);

  int j = 0;
  while (j < maxNeighbors)
  {
    int jidx = neighborList[j*nAtom + idx];

    // Uncoalesced read
    POSVECTYPE jpos = position[jidx];

    // Calculate distance
    FPTYPE delx = ipos.x() - jpos.x();
    FPTYPE dely = ipos.y() - jpos.y();
    FPTYPE delz = ipos.z() - jpos.z();
    FPTYPE r2inv = delx*delx + dely*dely + delz*delz;

    // If distance is less than cutoff, calculate force
    if (r2inv > 0 && r2inv < cutsq_t)
    {
      r2inv = (FPTYPE)1.0 / r2inv;
      FPTYPE r6inv = r2inv * r2inv * r2inv;
      FPTYPE forceC = r2inv * r6inv * (lj1_t * r6inv - lj2_t);

      f.x() += delx * forceC;
      f.y() += dely * forceC;
      f.z() += delz * forceC;
    }
    j++;
  }
  force[idx] = f;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "Usage: ./" << argv[0] << " <class size> <iteration>\n";
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
    position[i].x() = rand() % domainEdge;
    position[i].y() = rand() % domainEdge;
    position[i].z() = rand() % domainEdge;
  }

  std::cout << "Finished.\n";
  int totalPairs = buildNeighborList<FPTYPE, POSVECTYPE>(nAtom, position, neighborList);
  std::cout << totalPairs << " of " << nAtom*maxNeighbors
            << " pairs within cutoff distance = "
            << 100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %\n";

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);

  buffer<FORCEVECTYPE, 1> d_force(nAtom);
  buffer<POSVECTYPE, 1> d_position(position, nAtom);
  buffer<int, 1> d_neighborList(neighborList, nAtom * maxNeighbors);

  range<1> lws (256);
  range<1> gws ((nAtom + 255) / 256 * 256);

  // warmup and result verification
  q.submit([&](handler& cgh) {
    auto force = d_force.get_access<sycl_discard_write>(cgh);
    auto position = d_position.get_access<sycl_read>(cgh);
    auto neighborList = d_neighborList.get_access<sycl_read>(cgh);
    cgh.parallel_for<class warmup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      md(item,
         position.get_pointer(), 
         force.get_pointer(), 
         neighborList.get_pointer(), 
         nAtom,
         maxNeighbors,
         lj1,
         lj2,
         cutsq);
    });
  });

  q.submit([&](handler& cgh) {
    auto acc = d_force.get_access<sycl_read>(cgh);
    cgh.copy(acc, force);
  }).wait();

  std::cout << "Performing Correctness Check (may take several minutes)\n";

  checkResults<FPTYPE, FORCEVECTYPE, POSVECTYPE>(force, position, neighborList, nAtom);

  for (int i = 0; i < iteration; i++)
  {
    q.submit([&](handler& cgh) {
      auto force = d_force.get_access<sycl_discard_write>(cgh);
      auto position = d_position.get_access<sycl_read>(cgh);
      auto neighborList = d_neighborList.get_access<sycl_read>(cgh);
      cgh.parallel_for<class run>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        md(item,
           position.get_pointer(), 
           force.get_pointer(), 
           neighborList.get_pointer(), 
           nAtom,
           maxNeighbors,
           lj1,
           lj2,
           cutsq);
      });
    });
  }
  q.wait();

  free(position);
  free(force);
  free(neighborList);

  return 0;
}
