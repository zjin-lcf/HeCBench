#include <cassert>
#include <cfloat>
#include <list>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "MD.h"


template <class T, class posVecType>
inline T distance(const posVecType* position, const int i, const int j);

template <class T>
inline void insertInOrder(std::list<T>& currDist, std::list<int>& currList,
        const int j, const T distIJ, const int maxNeighbors);

template <class T, class posVecType>
inline int buildNeighborList(const int nAtom, const posVecType* position,
        int* neighborList);

template <class T>
inline int populateNeighborList(std::list<T>& currDist,
        std::list<int>& currList, const int j, const int nAtom,
        int* neighborList);


// ****************************************************************************
// Function: checkResults
//
// Purpose:
//   Check device results against cpu results -- this is the CPU equivalent of
//
// Arguments:
//      d_force:   forces calculated on the device
//      position:  positions of atoms
//      neighList: atom neighbor list
//      nAtom:     number of atoms
// Returns:  true if results match, false otherwise
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
template <class T, class forceVecType, class posVecType>
bool checkResults(forceVecType* d_force, posVecType *position,
                  int *neighList, int nAtom)
{
    for (int i = 0; i < nAtom; i++)
    {
        posVecType ipos = position[i];
        forceVecType f = {0.f, 0.f, 0.f, 0.f};
        int j = 0;
        while (j < maxNeighbors)
        {
            int jidx = neighList[j*nAtom + i];
            posVecType jpos = position[jidx];
            // Calculate distance
            T delx = ipos.x - jpos.x;
            T dely = ipos.y - jpos.y;
            T delz = ipos.z - jpos.z;
            T r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {

                r2inv = 1.0f/r2inv;
                T r6inv = r2inv * r2inv * r2inv;
                T force = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x += delx * force;
                f.y += dely * force;
                f.z += delz * force;
            }
            j++;
        }
        // Check the results
        T diffx = (d_force[i].x - f.x) / d_force[i].x;
        T diffy = (d_force[i].y - f.y) / d_force[i].y;
        T diffz = (d_force[i].z - f.z) / d_force[i].z;
        T err = sqrt(diffx*diffx) + sqrt(diffy*diffy) + sqrt(diffz*diffz);
        if (err > (3.0 * EPSILON))
        {
            std::cout << "Test Failed, idx: " << i << " diff: " << err << "\n";
            std::cout << "f.x: " << f.x << " df.x: " << d_force[i].x << "\n";
            std::cout << "f.y: " << f.y << " df.y: " << d_force[i].y << "\n";
            std::cout << "f.z: " << f.z << " df.z: " << d_force[i].z << "\n";
            std::cout << "Test FAILED\n";
            return false;
        }
    }
    std::cout << "Test Passed\n";
    return true;
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
    POSVECTYPE*   position;
    FORCEVECTYPE* h_force;
    int* neighborList;

    position = (POSVECTYPE*) malloc(nAtom * sizeof(POSVECTYPE));
    h_force = (FORCEVECTYPE*) malloc(nAtom * sizeof(FORCEVECTYPE));
    neighborList = (int*) malloc(maxNeighbors * nAtom * sizeof(int));

    std::cout << "Initializing test problem (this can take several "
            "minutes for large problems).\n                   ";

    // Seed random number generator
    srand48(8650341L);

    // Initialize positions -- random distribution in cubic domain
    for (int i = 0; i < nAtom; i++)
    {
        position[i].x = (drand48() * domainEdge);
        position[i].y = (drand48() * domainEdge);
        position[i].z = (drand48() * domainEdge);
    }


    std::cout << "Finished.\n";
    int totalPairs = buildNeighborList<FPTYPE, POSVECTYPE>(nAtom, position, neighborList);
    std::cout << totalPairs << " of " << nAtom*maxNeighbors <<
            " pairs within cutoff distance = " <<
            100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %\n";

    // see MD.h
    FPTYPE lj1_t   = (FPTYPE) lj1;
    FPTYPE lj2_t   = (FPTYPE) lj2;
    FPTYPE cutsq_t = (FPTYPE) cutsq;


    #pragma omp target data map(to: position[0:nAtom], \
		                    neighborList[0:nAtom * maxNeighbors]) \
                            map(from: h_force[0:nAtom])
    for (int i = 0; i < iteration; i++) {
      #pragma omp target teams distribute parallel for simd thread_limit(256) 
      for (uint idx = 0; idx < nAtom; idx ++) {
          POSVECTYPE ipos = position[idx];
          FORCEVECTYPE f = {0.0f, 0.0f, 0.0f, 0.0f};

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
              if (r2inv < cutsq_t)
              {
                  r2inv = 1.0f/r2inv;
                  FPTYPE r6inv = r2inv * r2inv * r2inv;
                  FPTYPE forceC = r2inv*r6inv*(lj1_t*r6inv - lj2_t);

                  f.x += delx * forceC;
                  f.y += dely * forceC;
                  f.z += delz * forceC;
              }
              j++;
          }
          // store the results
          h_force[idx] = f;
      }
    }

    if (iteration == 1 && !checkResults<FPTYPE, FORCEVECTYPE, POSVECTYPE>(h_force, position,
            neighborList, nAtom))
    {
        return -1;
    }

    free(position);
    free(h_force);
    free(neighborList);

    return 0;
}

// ********************************************************
// Function: distance
//
// Purpose:
//   Calculates distance squared between two atoms
//
// Arguments:
//   position: atom position information
//   i, j: indexes of the two atoms
//
// Returns:  the computed distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T, class posVecType>
inline T distance(const posVecType* position, const int i, const int j)
{
    posVecType ipos = position[i];
    posVecType jpos = position[j];
    T delx = ipos.x - jpos.x;
    T dely = ipos.y - jpos.y;
    T delz = ipos.z - jpos.z;
    T r2inv = delx * delx + dely * dely + delz * delz;
    return r2inv;
}

// ********************************************************
// Function: insertInOrder
//
// Purpose:
//   Adds atom j to current neighbor list and distance list
//   if it's distance is low enough.
//
// Arguments:
//   currDist: distance between current atom and each of its neighbors in the
//             current list, sorted in ascending order
//   currList: neighbor list for current atom, sorted by distance in asc. order
//   j:        atom to insert into neighbor list
//   distIJ:   distance between current atom and atom J
//   maxNeighbors: max length of neighbor list
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T>
inline void insertInOrder(std::list<T>& currDist, std::list<int>& currList,
        const int j, const T distIJ, const int maxNeighbors)
{

    typename std::list<T>::iterator   it;
    typename std::list<int>::iterator it2;

    it2 = currList.begin();

    T currMax = currDist.back();

    if (distIJ > currMax) return;

    for (it=currDist.begin(); it!=currDist.end(); it++)
    {
        if (distIJ < (*it))
        {
            // Insert into appropriate place in list
            currDist.insert(it,distIJ);
            currList.insert(it2, j);

            // Trim end of list
            currList.resize(maxNeighbors);
            currDist.resize(maxNeighbors);
            return;
        }
        it2++;
    }
}
// ********************************************************
// Function: buildNeighborList
//
// Purpose:
//   Builds the neighbor list structure for all atoms for GPU coalesced reads
//   and counts the number of pairs within the cutoff distance, so
//   the benchmark gets an accurate FLOPS count
//
// Arguments:
//   nAtom:    total number of atoms
//   position: pointer to the atom's position information
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T, class posVecType>
inline int buildNeighborList(const int nAtom, const posVecType* position,
        int* neighborList)
{
    int totalPairs = 0;
    // Build Neighbor List
    // Find the nearest N atoms to each other atom, where N = maxNeighbors
    for (int i = 0; i < nAtom; i++)
    {
        // Current neighbor list for atom i, initialized to -1
        std::list<int>   currList(maxNeighbors, -1);
        // Distance to those neighbors.  We're populating this with the
        // closest neighbors, so initialize to FLT_MAX
        std::list<T> currDist(maxNeighbors, FLT_MAX);

        for (int j = 0; j < nAtom; j++)
        {
            if (i == j) continue; // An atom cannot be its own neighbor

            // Calculate distance and insert in order into the current lists
            T distIJ = distance<T, posVecType>(position, i, j);
            insertInOrder<T>(currDist, currList, j, distIJ, maxNeighbors);
        }
        // We should now have the closest maxNeighbors neighbors and their
        // distances to atom i. Populate the neighbor list data structure
        // for GPU coalesced reads.
        // The populate method returns how many of the maxNeighbors closest
        // neighbors are within the cutoff distance.  This will be used to
        // calculate GFLOPS later.
        totalPairs += populateNeighborList<T>(currDist, currList, i, nAtom,
                neighborList);
    }
    return totalPairs;
}


// ********************************************************
// Function: populateNeighborList
//
// Purpose:
//   Populates the neighbor list structure for a *single* atom for
//   GPU coalesced reads and counts the number of pairs within the cutoff
//   distance, (for current atom) so the benchmark gets an accurate FLOPS count
//
// Arguments:
//   currDist: distance between current atom and each of its maxNeighbors
//             neighbors
//   currList: current list of neighbors
//   i:        current atom
//   nAtom:    total number of atoms
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T>
inline int populateNeighborList(std::list<T>& currDist,
        std::list<int>& currList, const int i, const int nAtom,
        int* neighborList)
{
    int idx = 0;
    int validPairs = 0; // Pairs of atoms closer together than the cutoff

    // Iterate across distance and neighbor list
    typename std::list<T>::iterator distanceIter = currDist.begin();
    for (std::list<int>::iterator neighborIter = currList.begin();
            neighborIter != currList.end(); neighborIter++)
    {
        // Populate packed neighbor list
        neighborList[(idx * nAtom) + i] = *neighborIter;

        // If the distance is less than cutoff, increment valid counter
        if (*distanceIter < cutsq)
            validPairs++;

        // Increment idx and distance iterator
        idx++;
        distanceIter++;
    }
    return validPairs;
}
