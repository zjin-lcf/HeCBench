#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>
#include "vectypes.h"

typedef float2 Float2;
typedef gmx::BasicVector<float> Float3;
typedef float4 Float4;

#include "constants.h"

inline __device__ void operator+=(float4 &a, float4 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

static __forceinline__  __device__
float pmeCorrF(const float z2)
{
  constexpr float FN6 = -1.7357322914161492954e-8F;
  constexpr float FN5 = 1.4703624142580877519e-6F;
  constexpr float FN4 = -0.000053401640219807709149F;
  constexpr float FN3 = 0.0010054721316683106153F;
  constexpr float FN2 = -0.019278317264888380590F;
  constexpr float FN1 = 0.069670166153766424023F;
  constexpr float FN0 = -0.75225204789749321333F;

  constexpr float FD4 = 0.0011193462567257629232F;
  constexpr float FD3 = 0.014866955030185295499F;
  constexpr float FD2 = 0.11583842382862377919F;
  constexpr float FD1 = 0.50736591960530292870F;
  constexpr float FD0 = 1.0F;

  const float z4 = z2 * z2;
        float polyFD0 = FD4 * z4 + FD2;
  const float polyFD1 = FD3 * z4 + FD1;
  polyFD0 = polyFD0 * z4 + FD0;
  polyFD0 = polyFD1 * z2 + polyFD0;
  polyFD0 = 1.0F / polyFD0;

  float polyFN0 = FN6 * z4 + FN4;
  float polyFN1 = FN5 * z4 + FN3;
  polyFN0       = polyFN0 * z4 + FN2;
  polyFN1       = polyFN1 * z4 + FN1;
  polyFN0       = polyFN0 * z4 + FN0;
  polyFN0       = polyFN1 * z2 + polyFN0;

  return polyFN0 * polyFD0;
}

static __forceinline__ __device__
void reduceForceIAndFShiftXYZ(
    float* sm_buf,
    const float* fCiBufX,
    const float* fCiBufY,
    const float* fCiBufZ,
    const bool   calcFShift,
    const int    tidxi,
    const int    tidxj,
    const int    sci,
    const int    shift,
    Float3 *a_f,
    Float3 *a_fShift) 
{
  static constexpr int bufStride  = c_clSize * c_clSize;
  static constexpr int clSizeLog2 = StaticLog2<c_clSize>::value;
  const int            tidx       = tidxi + tidxj * c_clSize;
  float                fShiftBuf  = 0.0F;
#pragma unroll 8
  for (int ciOffset = 0; ciOffset < c_nbnxnGpuNumClusterPerSupercluster; ciOffset++)
  {
    const int aidx = (sci * c_nbnxnGpuNumClusterPerSupercluster + ciOffset) * c_clSize + tidxi;
    /* store i forces in shmem */
    sm_buf[tidx]                 = fCiBufX[ciOffset];
    sm_buf[bufStride + tidx]     = fCiBufY[ciOffset];
    sm_buf[2 * bufStride + tidx] = fCiBufZ[ciOffset];
    __syncthreads();

    /* Reduce the initial c_clSize values for each i atom to half
     * every step by using c_clSize * i threads. */
    int i = c_clSize / 2;
    for (int j = clSizeLog2 - 1; j > 0; j--)
    {
      if (tidxj < i)
      {
        sm_buf[tidx] += sm_buf[tidx + i * c_clSize];
        sm_buf[bufStride + tidx] += sm_buf[bufStride + tidx + i * c_clSize];
        sm_buf[2 * bufStride + tidx] += sm_buf[2 * bufStride + tidx + i * c_clSize];
      }
      i >>= 1;
      __syncthreads();
    }

    /* i == 1, last reduction step, writing to global mem */
    /* Split the reduction between the first 3 line threads
       Threads with line id 0 will do the reduction for (float3).x components
       Threads with line id 1 will do the reduction for (float3).y components
       Threads with line id 2 will do the reduction for (float3).z components. */
    if (tidxj < 3)
    {
      const float f = sm_buf[tidxj * bufStride + tidxi] + 
                      sm_buf[tidxj * bufStride + c_clSize + tidxi];

      atomicAdd(&a_f[aidx][tidxj], f);

      if (calcFShift) fShiftBuf += f;
    }
    __syncthreads();
  }
  /* add up local shift forces into global mem */
  if (calcFShift)
  {
    /* Only threads with tidxj < 3 will update fshift.
       The threads performing the update must be the same as the threads
       storing the reduction result above. */
    if (tidxj < 3)
    {
      if constexpr (c_clSize == 4)
      {
        fShiftBuf += __shfl_down(fShiftBuf, 1);
        fShiftBuf += __shfl_down(fShiftBuf, 2);
        if (tidxi == 0)
        {
          atomicAdd(&a_fShift[shift][tidxj], fShiftBuf);
        }
      }
      else
      {
        atomicAdd(&a_fShift[shift][tidxj], fShiftBuf);
      }
    }
  }
}

static __forceinline__ __device__
void reduceForceJShuffle(
    Float3 f,
    const int tidxi,
    const int aidx,
    Float3 *a_f)
{
  static_assert(c_clSize == 8 || c_clSize == 4);
  f[0] += __shfl_down(f[0], 1);
  f[1] += __shfl_up(f[1], 1);
  f[2] += __shfl_down(f[2], 1);
  if (tidxi & 1)
  {
    f[0] = f[1];
  }

  f[0] += __shfl_down(f[0], 2);
  f[2] += __shfl_up(f[2], 2);
  if (tidxi & 2)
  {
    f[0] = f[2];
  }

  if constexpr (c_clSize == 8)
  {
    f[0] += __shfl_down(f[0], 4);
  }

  if (tidxi < 3)
  {
    atomicAdd(&a_f[aidx][tidxi], f[0]);
  }
}

static __forceinline__ __device__
void reduceForceJ(
    float* sm_buf,
    Float3 f,
    const int tidxi,
    const int tidxj,
    const int aidx,
    Float3 *a_f)
{
  reduceForceJShuffle(f, tidxi, aidx, a_f);
}

__global__ void nbnxmKernelTest(
    const Float4 *__restrict__ a_xq,
    Float3 *__restrict__ a_f,
    Float3 *__restrict__ a_shiftVec,
    Float3 *__restrict__ a_fShift,
    nbnxn_cj4_t *__restrict__ a_plistCJ4,
    const nbnxn_sci_t *__restrict__ a_plistSci,
    const nbnxn_excl_t *__restrict__ a_plistExcl,
    const int *__restrict__ a_atomTypes,
    const Float2 *__restrict__ a_nbfp,
    const int numTypes,
    const float rCoulombSq,
    const float ewaldBeta,
    const float epsFac,
    const bool calcShift) 
{
  constexpr int prunedClusterPairSize = c_clSize * c_splitClSize;

  /* thread/block/warp id-s */
  const unsigned tidxi = threadIdx.x;
  const unsigned tidxj = threadIdx.y;
  const unsigned tidx  = tidxj * c_clSize + tidxi;
  const unsigned bidx = blockIdx.z;

  const unsigned imeiIdx = tidx / prunedClusterPairSize;

  constexpr size_t local_mem_size =
    c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(Float4) + // sm_xq
    c_clSize * c_clSize * DIM * sizeof(float) +                       // sm_reductionBuffer
    c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(int);     // sm_atomTypeI

  __shared__ uint8_t localPtr[local_mem_size];
  uint8_t *ptr = localPtr; // localPtr is read only

  Float4* sm_xq = reinterpret_cast<Float4*>(ptr);
  ptr += c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(Float4);
  float* sm_reductionBuffer = reinterpret_cast<float*>(ptr);
  ptr += c_clSize * c_clSize * DIM * sizeof(float);
  int* sm_atomTypeI = reinterpret_cast<int*>(ptr);

  float fCiBufX[c_nbnxnGpuNumClusterPerSupercluster] = {0};
  float fCiBufY[c_nbnxnGpuNumClusterPerSupercluster] = {0};
  float fCiBufZ[c_nbnxnGpuNumClusterPerSupercluster] = {0};

  const nbnxn_sci_t nbSci     = a_plistSci[bidx];
  const int         sci       = nbSci.sci;
  const int         cij4Start = nbSci.cj4_ind_start;
  const int         cij4End   = nbSci.cj4_ind_end;
  const int         nbScishift = nbSci.shift;

  // Only needed if props.elecEwaldAna
  const float beta2 = ewaldBeta * ewaldBeta;
  const float beta3 = ewaldBeta * ewaldBeta * ewaldBeta;

  for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i += c_clSize)
  {
    /* Pre-load i-atom x and q into shared memory */
    const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj + i;
    const int ai = ci * c_clSize + tidxi;

    const Float3 shift = a_shiftVec[nbScishift];
    Float4       xqi = a_xq[ai];
    xqi += make_float4(shift[0], shift[1], shift[2], 0.0F);
    xqi.w *= epsFac;
    sm_xq[(tidxj + i) * c_clSize + tidxi] = xqi;
    sm_atomTypeI[(tidxj + i) * c_clSize + tidxi] = a_atomTypes[ai];
  }
  __syncthreads();

  // Only needed if (doExclusionForces)
  const bool nonSelfInteraction = !(nbScishift == c_centralShiftIndex & tidxj <= tidxi);

  // loop over the j clusters = seen by any of the atoms in the current super-cluster
  for (int j4 = cij4Start; j4 < cij4End; j4 += 1)
  {
    unsigned imask = a_plistCJ4[j4].imei[imeiIdx].imask;
    if (!imask)
    {
      continue;
    }
    const int wexclIdx = a_plistCJ4[j4].imei[imeiIdx].excl_ind;
    const unsigned wexcl = a_plistExcl[wexclIdx].pair[tidx & (prunedClusterPairSize - 1)];

    for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
    {
      const bool maskSet =
        imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster));
      if (!maskSet)
      {
        continue;
      }
      unsigned  maskJI = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
      const int cj     = a_plistCJ4[j4].cj[jm];
      const int aj     = cj * c_clSize + tidxj;

      // load j atom data
      const Float4 xqj = a_xq[aj];

      const Float3 xj(xqj.x, xqj.y, xqj.z);
      const float  qj = xqj.w;
      int          atomTypeJ; // Only needed if (!props.vdwComb)
      atomTypeJ = a_atomTypes[aj];

      Float3 fCjBuf(0.0F, 0.0F, 0.0F);

#pragma unroll 8
      for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
      {
        if (imask & maskJI)
        {
          // i cluster index
          const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + i;
          // all threads load an atom from i cluster ci into shmem!
          const Float4 xqi = sm_xq[i * c_clSize + tidxi];
          const Float3 xi(xqi.x, xqi.y, xqi.z);

          // distance between i and j atoms
          const Float3 rv = xi - xj;
          float        r2 = norm2(rv);

          const float pairExclMask = (wexcl & maskJI) ? 1.0F : 0.0F;

          // cutoff & exclusion check

          const bool notExcluded = (nonSelfInteraction | (ci != cj));

          // Check optimal way of branching here.
          if ((r2 < rCoulombSq) && notExcluded)
          {
            const float qi = xqi.w;
            int         atomTypeI; // Only needed if (!props.vdwComb)
            Float2      c6c12;

            /* LJ 6*C6 and 12*C12 */
            atomTypeI = sm_atomTypeI[i * c_clSize + tidxi];
            c6c12     = a_nbfp[numTypes * atomTypeI + atomTypeJ];

            // c6 and c12 are unused and garbage iff props.vdwCombLB && !doCalcEnergies
            const float c6  = c6c12.x;
            const float c12 = c6c12.y;

            // Ensure distance do not become so small that r^-12 overflows
            r2 = max(r2, c_nbnxnMinDistanceSquared);

            const float rInv = __frsqrt_rn(r2);
            const float r2Inv = rInv * rInv;
            float       r6Inv, fInvR;
            r6Inv = r2Inv * r2Inv * r2Inv;
            r6Inv *= pairExclMask;
            fInvR = r6Inv * (c12 * r6Inv - c6) * r2Inv;

            fInvR += qi * qj
              * (pairExclMask * r2Inv * rInv + pmeCorrF(beta2 * r2) * beta3);

            const Float3 forceIJ = rv * fInvR;

            /* accumulate j forces in registers */
            fCjBuf -= forceIJ;
            /* accumulate i forces in registers */
            fCiBufX[i] += forceIJ[0];
            fCiBufY[i] += forceIJ[1];
            fCiBufZ[i] += forceIJ[2];
          } // (r2 < rCoulombSq) && notExcluded
        }     // (imask & maskJI)
        /* shift the mask bit by 1 */
        maskJI += maskJI;
      } // for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
      /* reduce j forces */
      reduceForceJ(sm_reductionBuffer, fCjBuf, tidxi, tidxj, aj, a_f);
    } // for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
  } // for (int j4 = cij4Start; j4 < cij4End; j4 += 1)

  {
    const nbnxn_sci_t nbSci = a_plistSci[bidx];
    const int sci = nbSci.sci;

    // skip central shifts when summing shift forces
    const bool doCalcShift = (calcShift && nbSci.shift != c_centralShiftIndex);

    reduceForceIAndFShiftXYZ(
        sm_reductionBuffer, fCiBufX, fCiBufY, fCiBufZ, doCalcShift, tidxi, tidxj, sci, nbSci.shift, a_f, a_fShift);
  }
}

nbnxn_cj4_t get_cj4(int id) {
  nbnxn_cj4_t value;
  for (int i = 0; i < c_nbnxnGpuJgroupSize; ++i) {
    value.cj[i] = i + id;
  }
  for (int i = 0; i < c_nbnxnGpuClusterpairSplit; ++i) {
    value.imei[i].imask = 0U;
    value.imei[i].excl_ind = 0;
  }
  return value;
}

nbnxn_sci_t get_sci(int id) {
  return {id, 0, 8 * id, 8 * id + 7};
}

nbnxn_excl_t get_excl(int id) {
  nbnxn_excl_t value;
  for (int i = 0; i < c_nbnxnGpuExclSize; ++i) {
    value.pair[i] = 7;
  }
  return value;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const dim3 blocks ( block_x, block_y, 1 );
  const dim3 grids ( 1, 1, grid_z );

  Float4* a_xq;
  hipMallocManaged(&a_xq, sizeof(Float4) * NUM_ATOMS);

  Float3* a_f;
  hipMallocManaged(&a_f, sizeof(Float3) * NUM_ATOMS);

  Float3* shiftVec;
  hipMallocManaged(&shiftVec, sizeof(Float3) * 45);

  Float3* fShift;
  hipMallocManaged(&fShift, sizeof(Float3) * 45);

  nbnxn_cj4_t* cj4;
  hipMallocManaged(&cj4, sizeof(nbnxn_cj4_t) * 56881);

  nbnxn_sci_t* sci;
  hipMallocManaged(&sci, sizeof(nbnxn_sci_t) * 4806);

  nbnxn_excl_t* excl;
  hipMallocManaged(&excl, sizeof(nbnxn_excl_t) * 19205);

  int* atomTypes;
  hipMallocManaged(&atomTypes, sizeof(int) * NUM_ATOMS);

  Float2* nbfp;
  hipMallocManaged(&nbfp, sizeof(Float2) * 1024);

  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_xq[i] = make_float4(1.0f, 0.5f, 0.25f, 0.125f);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_f[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    shiftVec[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    fShift[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 56881; ++i) {
    cj4[i] = get_cj4(i);
  }
  for (int i = 0; i < 4806; ++i) {
    sci[i] = get_sci(i);
  }
  for (int i = 0; i < 19205; ++i) {
    excl[i] = get_excl(i);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    atomTypes[i] = (i % 2);
  }
  for (int i = 0; i < 1024; ++i) {
    nbfp[i] = make_float2(0.5f, 0.25f);
  }

  // Warming-up
  hipLaunchKernelGGL(nbnxmKernelTest, grids, blocks, 0, 0, 
      a_xq,
      a_f,
      shiftVec,
      fShift,
      cj4,
      sci,
      excl,
      atomTypes,
      nbfp,
      32,
      1,
      3.12341,
      138.935,
      0);
  hipDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    hipLaunchKernelGGL(nbnxmKernelTest, grids, blocks, 0, 0, 
        a_xq,
        a_f,
        shiftVec,
        fShift,
        cj4,
        sci,
        excl,
        atomTypes,
        nbfp,
        32,
        1,
        3.12341,
        138.935,
        0);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (w/o shift): %f (us)\n", (time * 1e-3f) / repeat);

#ifdef DEBUG
  float f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < NUM_ATOMS; ++i) {
    f0 += a_f[i][0];
    f1 += a_f[i][1];
    f2 += a_f[i][2];
  }
  printf("Checksum (a_f): %f %f %f\n", f0, f1, f2);

  f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < 45; ++i) {
    f0 += fShift[i][0];
    f1 += fShift[i][1];
    f2 += fShift[i][2];
  }
  printf("Checksum (fShift): %f %f %f\n", f0, f1, f2);
#endif

  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_xq[i] = make_float4(1.0f, 0.5f, 0.25f, 0.125f);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_f[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    shiftVec[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    fShift[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 56881; ++i) {
    cj4[i] = get_cj4(i);
  }
  for (int i = 0; i < 4806; ++i) {
    sci[i] = get_sci(i);
  }
  for (int i = 0; i < 19205; ++i) {
    excl[i] = get_excl(i);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    atomTypes[i] = (i % 2);
  }
  for (int i = 0; i < 1024; ++i) {
    nbfp[i] = make_float2(0.5f, 0.25f);
  }

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    hipLaunchKernelGGL(nbnxmKernelTest, grids, blocks, 0, 0, 
        a_xq,
        a_f,
        shiftVec,
        fShift,
        cj4,
        sci,
        excl,
        atomTypes,
        nbfp,
        32,
        1,
        3.12341,
        138.935,
        1);  // compute shift
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (w/ shift): %f (us)\n", (time * 1e-3f) / repeat);

#ifdef DEBUG
  f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < NUM_ATOMS; ++i) {
    f0 += a_f[i][0];
    f1 += a_f[i][1];
    f2 += a_f[i][2];
  }
  printf("Checksum (a_f): %f %f %f\n", f0, f1, f2);

  f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < 45; ++i) {
    f0 += fShift[i][0];
    f1 += fShift[i][1];
    f2 += fShift[i][2];
  }
  printf("Checksum (fShift): %f %f %f\n", f0, f1, f2);
#endif

  hipFree(nbfp);
  hipFree(atomTypes);
  hipFree(excl);
  hipFree(sci);
  hipFree(cj4);
  hipFree(fShift);
  hipFree(shiftVec);
  hipFree(a_f);
  hipFree(a_xq);

  return 0;
}
