#include <iostream>
#include <vector>
#include "mttkrp_gpu.h"

template <typename T>
inline void atomicAdd(T &var, T val) 
{
  auto atm = sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(var);
  atm.fetch_add(val);
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar(
  const DTYPE *__restrict vals,
  const ITYPE *__restrict fbrLikeSlcInds,
  const ITYPE *__restrict dInds2,
  const ITYPE *__restrict fbrPtr0,
  const ITYPE *__restrict fbrPtr1,
  const ITYPE *__restrict fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict dU0,
  const DTYPE *__restrict dU1,
  const DTYPE *__restrict dU2,
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW,
  sycl::nd_item<1> &item)
{
  ITYPE tId = item.get_local_id(0);
  ITYPE laneId = tId & 31;
  ITYPE bdim = item.get_local_range(0);
  ITYPE gId = (item.get_group(0) * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = (gId >> (5 + logOfWPC)) << logOfFPW; // 5: minimum 1 WARP (2^5) // item.get_group(0) ;//

  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){

    tmp_val = 0;
    bool diffFiber = false;
    unsigned int idx0;

    for (int fr = 0; fr < fbrPerWarp && (fbr+fr) < (nFibers - 1); ++fr){

      diffFiber = false;
      unsigned int idx1 = fbrIdx1[fbr+fr];// dInds1[fbrPtr1[fbr]];
      idx0 = fbrLikeSlcInds[fbr+fr];//slc;
      tmp_val = 0;

      for(unsigned int x = fbrPtr1[fbr+fr] + workId; x < fbrPtr1[fbr+fr+1]; x+=warpPerSlice) {

        unsigned int idx2 = dInds2[x];

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp_val += vals[x] * dU2[idx2 * R + r]; //2MR
        }
      }

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp += tmp_val * dU1[idx1 * R + r] ; //2PR
      }

      if(fbrLikeSlcInds[fbr+fr] != fbrLikeSlcInds[fbr+fr+1]) {

        diffFiber = true;
        for(unsigned int r=laneId; r<R; r+=32) {
          atomicAdd(dU0[idx0 * R + r], tmp); //2PR
        }
        tmp = 0;
      }
    }

    if(!diffFiber) {
      for(unsigned int r=laneId; r<R; r+=32) {
        atomicAdd(dU0[idx0 * R + r], tmp);
      }
    }
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict vals,
  const ITYPE *__restrict fbrLikeSlcInds,
  const ITYPE *__restrict dInds3,
  const ITYPE *__restrict fbrPtr0,
  const ITYPE *__restrict fbrPtr1,
  const ITYPE *__restrict fbrIdx1,
  const ITYPE *__restrict fbrPtr2,
  const ITYPE *__restrict fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict dU0,
  const DTYPE *__restrict dU1,
  const DTYPE *__restrict dU2,
  const DTYPE *__restrict dU3,
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW,
  sycl::nd_item<1> &item)
{
  ITYPE tId = item.get_local_id(0);
  ITYPE laneId = tId & 31;
  ITYPE bdim = item.get_local_range(0);
  ITYPE gId = (item.get_group(0) * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = (gId >> (5 + logOfWPC)) << logOfFPW; // 5: minimum 1 WARP (2^5) // item.get_group(0) ;//
  DTYPE tmp = 0, tmp_val, tmp2= 0;

  if(fbrS < nFibers - 1){

    tmp_val = 0;
    bool diffFiber = false;
    unsigned int idx0;

    for (int fr = 0; fr < fbrPerWarp && (fbrS+fr) < (nFibers - 1); ++fr){

      diffFiber = false;
      unsigned int idx1 = fbrIdx1[fbrS+fr];// dInds1[fbrPtr1[fbr]];
      idx0 = fbrLikeSlcInds[fbrS+fr];//slc;
      tmp = 0;

      for (int fbr = fbrPtr1[fbrS+fr] + workId; fbr < fbrPtr1[fbrS+fr+1]; fbr+=warpPerSlice){
        ITYPE idx2 = fbrIdx2[fbr];
        tmp_val = 0;

        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; x++) {

          unsigned int idx3 = dInds3[x];

          for(unsigned int r=laneId; r<R; r+=32) {
            tmp_val += vals[x] * dU3[idx3 * R + r]; //2MR
          }
        }

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp += tmp_val * dU2[idx2 * R + r] ;
        }
      }
      for(unsigned int r=laneId; r<R; r+=32) {
        tmp2 += tmp * dU1[idx1 * R + r] ;
      }

      if(fbrLikeSlcInds[fbrS+fr] != fbrLikeSlcInds[fbrS+fr+1]) {

        diffFiber = true;
        for(unsigned int r=laneId; r<R; r+=32) {
          atomicAdd(dU0[idx0 * R + r], tmp2); //2PR
        }
        tmp2 = 0;
      }
    }

    if(!diffFiber) {
      for(unsigned int r=laneId; r<R; r+=32) {
        atomicAdd(dU0[idx0 * R + r], tmp2); //2PR
      }
    }
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict vals,
  const ITYPE *__restrict fbrLikeSlcInds,
  const ITYPE *__restrict dInds3,
  const ITYPE *__restrict fbrPtr0,
  const ITYPE *__restrict fbrPtr1,
  const ITYPE *__restrict fbrIdx1,
  const ITYPE *__restrict fbrPtr2,
  const ITYPE *__restrict fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict dU0,
  const DTYPE *__restrict dU1,
  const DTYPE *__restrict dU2,
  const DTYPE *__restrict dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  sycl::nd_item<1> &item)
{
  ITYPE tId = item.get_local_id(0);
  ITYPE laneId = tId & 31;
  ITYPE bdim = item.get_local_range(0);
  ITYPE gId = (item.get_group(0) * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // item.get_group(0) ;//
  DTYPE tmp = 0, tmp_val, tmp2 = 0;

  if(fbrS < nFibers - 1){

    tmp = 0;
    unsigned int idx0 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];
    unsigned int idx3 = fbrLikeSlcInds[fbrS];//slc;

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      unsigned int idx1 = fbrIdx2[fbr];
      tmp_val = 0;

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx2 = dInds3[x];

        for(unsigned int r=laneId; r<R; r+=32)
          tmp_val += vals[x] * dU2[idx2 * R + r] ; //2MR
      }
      for(unsigned int r=laneId; r<R; r+=32)
        tmp += tmp_val * dU1[idx1 * R + r]  ;
    }
    for(unsigned int r=laneId; r<R; r+=32) {
      tmp2 = tmp * dU3[idx3 * R + r];
      atomicAdd(dU0[idx0 * R + r], tmp2); //2PR
    }
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar(
  const DTYPE *__restrict vals,
  const ITYPE *__restrict fbrLikeSlcInds,
  const ITYPE *__restrict dInds2,
  const ITYPE *__restrict fbrPtr0,
  const ITYPE *__restrict fbrPtr1,
  const ITYPE *__restrict fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict dU0,
  const DTYPE *__restrict dU1,
  const DTYPE *__restrict dU2,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  sycl::nd_item<1> &item)
{
  ITYPE tId = item.get_local_id(0);
  ITYPE laneId = tId & 31;
  ITYPE bdim = item.get_local_range(0);
  ITYPE gId = (item.get_group(0) * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // item.get_group(0) ;//
  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){

    tmp_val = 0;
    unsigned int idx0 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];
    unsigned int idx2 = fbrLikeSlcInds[fbr];//slc;

    for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

      unsigned int idx1 = dInds2[x];

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp_val += vals[x] * dU1[idx1 * R + r]; //2MR
      }
    }
    for(unsigned int r=laneId; r<R; r+=32) {
      tmp = tmp_val * dU2[idx2 * R + r] ;
      atomicAdd(dU0[idx0 * R + r], tmp); //2PR
    }
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict vals,
  const ITYPE *__restrict fbrLikeSlcInds,
  const ITYPE *__restrict dInds3,
  const ITYPE *__restrict fbrPtr0,
  const ITYPE *__restrict fbrPtr1,
  const ITYPE *__restrict fbrIdx1,
  const ITYPE *__restrict fbrPtr2,
  const ITYPE *__restrict fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict dU0,
  const DTYPE *__restrict dU1,
  const DTYPE *__restrict dU2,
  const DTYPE *__restrict dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  sycl::nd_item<1> &item)
{
  ITYPE tId = item.get_local_id(0);
  ITYPE laneId = tId & 31;
  ITYPE bdim = item.get_local_range(0);
  ITYPE gId = (item.get_group(0) * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // item.get_group(0) ;//
  DTYPE tmp;

  if(fbrS < nFibers - 1){

    unsigned int idx2 = fbrLikeSlcInds[fbrS];//slc;
    unsigned int idx3 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      unsigned int idx0 = fbrIdx2[fbr];
      tmp = 0;

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx1 = dInds3[x];

        for(unsigned int r=laneId; r<R; r+=32)
          tmp += vals[x] * dU1[idx1 * R + r]; //2MR
      }
      for(unsigned int r=laneId; r<R; r+=32)  {
        atomicAdd(dU0[idx0 * R + r], tmp * dU2[idx2 * R + r] * dU3[idx3 * R + r]) ;
      }
    }
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar(
  const DTYPE *__restrict vals,
  const ITYPE *__restrict fbrLikeSlcInds,
  const ITYPE *__restrict dInds2,
  const ITYPE *__restrict fbrPtr0,
  const ITYPE *__restrict fbrPtr1,
  const ITYPE *__restrict fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict dU0,
  const DTYPE *__restrict dU1,
  const DTYPE *__restrict dU2,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  sycl::nd_item<1> &item)
{
  ITYPE tId = item.get_local_id(0);
  ITYPE laneId = tId & 31;
  ITYPE bdim = item.get_local_range(0);
  ITYPE gId = (item.get_group(0) * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // item.get_group(0) ;//
  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){

    tmp_val = 0;
    unsigned int idx1 = fbrLikeSlcInds[fbr];//slc;
    unsigned int idx2 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];

    for(unsigned int r=laneId; r<R; r+=32)
      tmp = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR

    for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

      unsigned int idx0 = dInds2[x];

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp_val = vals[x] * tmp;///dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //2MR
        atomicAdd(dU0[idx0 * R + r], tmp_val);
      }
    }
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict vals,
  const ITYPE *__restrict fbrLikeSlcInds,
  const ITYPE *__restrict dInds3,
  const ITYPE *__restrict fbrPtr0,
  const ITYPE *__restrict fbrPtr1,
  const ITYPE *__restrict fbrIdx1,
  const ITYPE *__restrict fbrPtr2,
  const ITYPE *__restrict fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict dU0,
  const DTYPE *__restrict dU1,
  const DTYPE *__restrict dU2,
  const DTYPE *__restrict dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  sycl::nd_item<1> &item)
{
  ITYPE tId = item.get_local_id(0);
  ITYPE laneId = tId & 31;
  ITYPE bdim = item.get_local_range(0);
  ITYPE gId = (item.get_group(0) * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // item.get_group(0) ;//
  DTYPE tmp = 0, tmp_val = 0;;

  if(fbrS < nFibers - 1){

    tmp = 0;
    unsigned int idx1 = fbrLikeSlcInds[fbrS];//slc;
    unsigned int idx2 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];

    for(unsigned int r=laneId; r<R; r+=32)
      tmp_val = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      ITYPE idx3 = fbrIdx2[fbr];

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx0 = dInds3[x];

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp = vals[x] * dU3[idx3 * R + r] * tmp_val;//2MR
          atomicAdd(dU0[idx0 * R + r], tmp);
        }
      }
    }
  }
}


int MTTKRP_MIHCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){

  ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc = 0,  dFbrIdxLoc = 0, dFbrLoc2 = 0;
  ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0, totFbrIdx = 0, totFbrPtr2 = 0;

  // All m same mode
  ITYPE mode0 = 0;//TiledX[0].modeOrder[0];
  ITYPE mode1 = 1;//TiledX[0].modeOrder[1];
  ITYPE mode2 = 2;//TiledX[0].modeOrder[2];
  ITYPE mode3 = 3;//((TiledX[0].ndims == 4) ? TiledX[0].modeOrder[3] : 0) ;

  for (int m = 0; m < TiledX[0].ndims; ++m){

    if (TiledX[m].totNnz == 0) continue;

    totNnz += TiledX[m].totNnz;
    totSlcPtr += TiledX[m].fbrPtr[0].size() ;
    totSlcIdx += TiledX[m].fbrIdx[0].size() ;
    totFbrPtr += TiledX[m].fbrPtr[1].size() ;
    totFbrIdx += TiledX[m].fbrIdx[1].size() ;
    totFbrPtr2 += ((TiledX[m].ndims == 4) ? TiledX[m].fbrPtr[2].size() : 0) ;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Allocate Tensor on a device
  DTYPE* dVals = sycl::malloc_device<DTYPE>(totNnz, q);
  ITYPE* dFbrPtr0 = sycl::malloc_device<ITYPE>(totSlcPtr, q);
  ITYPE* dFbrIdx0 = sycl::malloc_device<ITYPE>(totSlcIdx, q);
  ITYPE* dFbrPtr1 = sycl::malloc_device<ITYPE>(totFbrPtr, q);
  ITYPE* dFbrIdx1 = sycl::malloc_device<ITYPE>(totFbrIdx, q);
  ITYPE* dFbrLikeSlcInds = sycl::malloc_device<ITYPE>(totFbrIdx, q);

  // conditional buffer allocation will produce undefined variables
  // ndim = 3
  ITYPE *dInds2, *dFbrIdx2, *dFbrPtr2, *dInds3;
  if(TiledX[0].ndims == 3)
    dInds2 = sycl::malloc_device<ITYPE>(totNnz, q);

  // ndim = 4
  if(TiledX[0].ndims == 4){
    dFbrIdx2 = sycl::malloc_device<ITYPE>(totFbrPtr2, q);
    dFbrPtr2 = sycl::malloc_device<ITYPE>(totFbrPtr2, q);
    dInds3 = sycl::malloc_device<ITYPE>(totNnz, q);
  }

  // device memory copy for tiled parts
  for (int m = 0; m < TiledX[0].ndims; ++m){

    if(m > 0) {

      if (TiledX[m-1].totNnz > 0) {

        dLoc += TiledX[m-1].totNnz;
        dSlcLoc += TiledX[m - 1].fbrPtr[0].size(); // all m same
        dSlcIdxLoc += TiledX[m - 1].fbrIdx[0].size();
        dFbrLoc += TiledX[m - 1].fbrPtr[1].size();
        dFbrIdxLoc += TiledX[m - 1].fbrIdx[1].size();
        dFbrLoc2 += ((TiledX[m].ndims == 4) ? TiledX[m - 1].fbrPtr[2].size() : 0) ;
      }
    }

    if (TiledX[m].totNnz == 0) continue;


    q.memcpy(dVals + dLoc, &(TiledX[m].vals[0]),
             TiledX[m].totNnz * sizeof(DTYPE));
    q.memcpy(dFbrPtr0 + dSlcLoc, &(TiledX[m].fbrPtr[0][0]),
             TiledX[m].fbrPtr[0].size() * sizeof(ITYPE));
    q.memcpy(dFbrIdx0 + dSlcIdxLoc, &(TiledX[m].fbrIdx[0][0]),
             TiledX[m].fbrIdx[0].size() * sizeof(ITYPE));
    q.memcpy(dFbrPtr1 + dFbrLoc, &(TiledX[m].fbrPtr[1][0]),
             TiledX[m].fbrPtr[1].size() * sizeof(ITYPE));
    q.memcpy(dFbrIdx1 + dFbrIdxLoc, &(TiledX[m].fbrIdx[1][0]),
             TiledX[m].fbrIdx[1].size() * sizeof(ITYPE));
    q.memcpy(dFbrLikeSlcInds + dFbrIdxLoc, &(TiledX[m].fbrLikeSlcInds[0]),
             TiledX[m].fbrIdx[1].size() * sizeof(ITYPE));

    if(TiledX[m].ndims == 3){
      if(m <= 2)
        q.memcpy(dInds2 + dLoc, &(TiledX[m].inds[TiledX[m].modeOrder[2]][0]),
                 TiledX[m].totNnz * sizeof(ITYPE));
    }
    if(TiledX[m].ndims == 4){
      q.memcpy(dFbrPtr2 + dFbrLoc2, &(TiledX[m].fbrPtr[2][0]),
               TiledX[m].fbrPtr[2].size() * sizeof(ITYPE));
      q.memcpy(dFbrIdx2 + dFbrLoc2, &(TiledX[m].fbrIdx[2][0]),
               TiledX[m].fbrIdx[2].size() * sizeof(ITYPE));
      q.memcpy(dInds3 + dLoc, &(TiledX[m].inds[TiledX[m].modeOrder[3]][0]),
               TiledX[m].totNnz * sizeof(ITYPE));
    }
  }

  //Matrices
  unsigned int *dULoc =  new unsigned int[TiledX[0].ndims];
  unsigned int *szDU =  new unsigned int[TiledX[0].ndims];

  //Matrices
  ITYPE mtxSize = ((TiledX[0].ndims == 3) ? (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows) * U[mode0].nCols
      : (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows + U[mode3].nRows) * U[mode0].nCols );

  DTYPE* dU = sycl::malloc_device<DTYPE>(mtxSize, q);

  for (int m = 0; m < TiledX[0].ndims; ++m)
    szDU[m] = U[m].nRows * U[m].nCols;

  q.memset(dU, 0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));

  q.memcpy(dU + szDU[0], &(U[mode1].vals[0]),
           U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE));
  q.memcpy(dU + szDU[0] + szDU[1], &(U[mode2].vals[0]),
           U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE));

  if(TiledX[0].ndims == 4)
    q.memcpy(dU + szDU[0] + szDU[1] + szDU[2], &(U[mode3].vals[0]),
             U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE));


  dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0, dFbrIdxLoc = 0, dFbrLoc2= 0;

  for (int MTTKRPmode = 0; MTTKRPmode < TiledX[0].ndims; ++MTTKRPmode){

    if(MTTKRPmode > 0){

      dLoc = 0; dSlcLoc = 0; dSlcIdxLoc = 0; dFbrLoc =0; dFbrIdxLoc = 0; dFbrLoc2= 0;

      // MTTKRP on mode mode 0 changed DU0. To pass correctness for now initializing to 2 again.
      int mode = MTTKRPmode - 1;
      for(long r = 0; r < U[mode].nRows; ++r){
        for(long c = 0; c < U[mode].nCols; ++c)
          U[mode].vals[r * U[mode].nCols + c] = mode + .5;
      }

      if(MTTKRPmode == 1){

        q.memcpy(dU, &(U[mode0].vals[0]), U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
        q.memset(dU + szDU[0], 0, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE));

      }
      else if(MTTKRPmode == 2){
        q.memcpy(dU + szDU[0], &(U[mode1].vals[0]),
                 U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE));
        q.memset(dU + szDU[0] + szDU[1], 0, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE));
      }
      else if(MTTKRPmode == 3){
        q.memcpy(dU + szDU[0] + szDU[1] , &(U[mode2].vals[0]),
              U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE));
        q.memset(dU + szDU[0] + szDU[1] + szDU[2], 0, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE));
      }
    }

    for (int m = 0; m < TiledX[0].ndims; ++m){

      /* matrix order according to mode order*/
      for (int mm = 0; mm < TiledX[0].ndims; ++mm){

        int curMode = TiledX[m].modeOrder[mm];
        dULoc[mm] = 0;

        for (int q = 0; q < curMode; ++q)
          dULoc[mm] +=  szDU[q % TiledX[0].ndims]; //1 2 3 0
      }

      if(m > 0) {

        if (TiledX[m-1].totNnz > 0) {

          dLoc += TiledX[m-1].totNnz;
          dSlcLoc += TiledX[m - 1].fbrPtr[0].size();
          dSlcIdxLoc += TiledX[m - 1].fbrIdx[0].size();
          dFbrLoc += TiledX[m - 1].fbrPtr[1].size();
          dFbrIdxLoc += TiledX[m - 1].fbrIdx[1].size();
          dFbrLoc2 += ((TiledX[0].ndims == 4) ? TiledX[m - 1].fbrPtr[2].size(): 0) ;
        }
      }

      if (TiledX[m].totNnz == 0) continue;

      int BLOCKSIZE;
      ITYPE opt_mode = Opt.mode;
      ITYPE opt_R = Opt.R;
      ITYPE tile_nFibers = TiledX[m].nFibers;

      if(TiledX[m].modeOrder[0] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: Slc atomics\n" ;

        BLOCKSIZE = Opt.TBsize;

        int warpPerFbr = Opt.warpPerSlice;//4;
        int logOfWarpPerFbr = log2(warpPerFbr);
        int fbrPerWarp = Opt.fiberPerWarp;//1;//BLOCKSIZE/32; // dont overflow TB
        int logOfFbrPerWarp = log2(fbrPerWarp );

        if( (warpPerFbr > (BLOCKSIZE/32)) || (fbrPerWarp > (BLOCKSIZE/32)) ){
          std::cout << "warpPerFbr (-w) or fbrPerWarp (-s) cannot be higher than threadblock size!"
            << std::endl << "hint: increase -b!" << std::endl;
          return -1;
        }

        const int grid = ( warpPerFbr * 32 * ((tile_nFibers + fbrPerWarp-1)/fbrPerWarp) + BLOCKSIZE - 1) / BLOCKSIZE;
        sycl::range<1> gws (grid * BLOCKSIZE);
        sycl::range<1> lws (BLOCKSIZE);

        if(TiledX[0].ndims == 3)
          q.submit([&] (sycl::handler &cgh) {
            auto dU_dULoc0 = dU + dULoc[0];
            auto dU_dULoc1 = dU + dULoc[1];
            auto dU_dULoc2 = dU + dULoc[2];
            cgh.parallel_for<class slc_atomic>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar(
                dVals + dLoc,
                dFbrLikeSlcInds + dFbrIdxLoc,
                dInds2 + dLoc,
                dFbrPtr0 + dSlcLoc,
                dFbrPtr1 + dFbrLoc,
                dFbrIdx1 + dFbrIdxLoc,
                tile_nFibers,  // TiledX[m].nFibers
                dU_dULoc0, //dU + dULoc[0],
                dU_dULoc1, //dU + dULoc[1],
                dU_dULoc2, //dU + dULoc[2],
                opt_mode, opt_R,
                warpPerFbr, logOfWarpPerFbr,
                fbrPerWarp, logOfFbrPerWarp,
                item);
            });
          });

        else if(TiledX[0].ndims == 4) {
          q.submit([&] (sycl::handler &cgh) {
            auto dU_dULoc0 = dU + dULoc[0];
            auto dU_dULoc1 = dU + dULoc[1];
            auto dU_dULoc2 = dU + dULoc[2];
            auto dU_dULoc3 = dU + dULoc[3];
            cgh.parallel_for<class slc_atomic_4d>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D(
                dVals + dLoc,
                dFbrLikeSlcInds + dFbrIdxLoc,
                dInds3 + dLoc,
                dFbrPtr0 + dSlcLoc,
                dFbrPtr1 + dFbrLoc,
                dFbrIdx1 + dFbrIdxLoc,
                dFbrPtr2 + dFbrLoc2,
                dFbrIdx2 + dFbrLoc2,
                tile_nFibers,
                dU_dULoc0,
                dU_dULoc1,
                dU_dULoc2,
                dU_dULoc3,
                opt_mode, opt_R,
                warpPerFbr, logOfWarpPerFbr,
                fbrPerWarp, logOfFbrPerWarp,
                item);
            });
          });
        }
      }

      else if(TiledX[0].ndims == 4 && TiledX[m].modeOrder[1] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: FbrS atomics\n";

        BLOCKSIZE = Opt.TBsize;

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        const int grid = ( warpPerFbr * 32 * tile_nFibers + BLOCKSIZE - 1) / BLOCKSIZE;
        sycl::range<1> gws (grid * BLOCKSIZE);
        sycl::range<1> lws (BLOCKSIZE);

        q.submit([&] (sycl::handler &cgh) {
          auto dU_dULoc0 = dU + dULoc[0];
          auto dU_dULoc1 = dU + dULoc[1];
          auto dU_dULoc2 = dU + dULoc[2];
          auto dU_dULoc3 = dU + dULoc[3];
          cgh.parallel_for<class fbrs_atomic>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D(
              dVals + dLoc,
              dFbrLikeSlcInds + dFbrIdxLoc,
              dInds3 + dLoc,
              dFbrPtr0 + dSlcLoc,
              dFbrPtr1 + dFbrLoc,
              dFbrIdx1 + dFbrIdxLoc,
              dFbrPtr2 + dFbrLoc2,
              dFbrIdx2 + dFbrLoc2,
              tile_nFibers,
              dU_dULoc1,
              dU_dULoc2,
              dU_dULoc3,
              dU_dULoc0,
              opt_mode, opt_R,
              warpPerFbr, logOfWarpPerFbr,
              item);
          });
        });
      }

      else if(TiledX[m].modeOrder[TiledX[0].ndims-2] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: Fbr atomics\n";

        BLOCKSIZE = Opt.TBsize;

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        int grid = ( warpPerFbr * 32 * tile_nFibers + BLOCKSIZE - 1) / BLOCKSIZE;
        sycl::range<1> gws (grid * BLOCKSIZE);
        sycl::range<1> lws (BLOCKSIZE);

        if(TiledX[0].ndims == 3)
          q.submit([&] (sycl::handler &cgh) {
            auto dU_dULoc0 = dU + dULoc[0];
            auto dU_dULoc1 = dU + dULoc[1];
            auto dU_dULoc2 = dU + dULoc[2];
            cgh.parallel_for<class fbr_atomic>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar(
                dVals + dLoc,
                dFbrLikeSlcInds + dFbrIdxLoc,
                dInds2 + dLoc,
                dFbrPtr0 + dSlcLoc,
                dFbrPtr1 + dFbrLoc,
                dFbrIdx1 + dFbrIdxLoc,
                tile_nFibers,
                //dU + dULoc[1],
                //dU + dULoc[2],
                //dU + dULoc[0],
                dU_dULoc1,
                dU_dULoc2,
                dU_dULoc0,
                opt_mode, opt_R,
                warpPerFbr, logOfWarpPerFbr,
                item);
            });
          });

        else if (TiledX[0].ndims == 4) {
          q.submit([&] (sycl::handler &cgh) {
            auto dU_dULoc0 = dU + dULoc[0];
            auto dU_dULoc1 = dU + dULoc[1];
            auto dU_dULoc2 = dU + dULoc[2];
            auto dU_dULoc3 = dU + dULoc[3];
            cgh.parallel_for<class fbr_atomic_4d>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D(
                dVals + dLoc,
                dFbrLikeSlcInds + dFbrIdxLoc,
                dInds3 + dLoc,
                dFbrPtr0 + dSlcLoc,
                dFbrPtr1 + dFbrLoc,
                dFbrIdx1 + dFbrIdxLoc,
                dFbrPtr2 + dFbrLoc2,
                dFbrIdx2 + dFbrLoc2,
                tile_nFibers,
                dU_dULoc2,
                dU_dULoc3,
                dU_dULoc0,
                dU_dULoc1,
                opt_mode, opt_R,
                warpPerFbr, logOfWarpPerFbr,
                item);
            });
          });
        }
      }

      else if(TiledX[m].modeOrder[TiledX[0].ndims-1] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: nnz atomics\n";

        BLOCKSIZE = Opt.TBsize;

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        const int grid = ( warpPerFbr * 32 * tile_nFibers + BLOCKSIZE - 1) / BLOCKSIZE;
        sycl::range<1> gws (grid * BLOCKSIZE);
        sycl::range<1> lws (BLOCKSIZE);

        if (TiledX[0].ndims == 3)
          q.submit([&] (sycl::handler &cgh) {
            auto dU_dULoc0 = dU + dULoc[0];
            auto dU_dULoc1 = dU + dULoc[1];
            auto dU_dULoc2 = dU + dULoc[2];
            cgh.parallel_for<class nnz_atomic>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar(
                dVals + dLoc,
                dFbrLikeSlcInds + dFbrIdxLoc,
                dInds2 + dLoc,
                dFbrPtr0 + dSlcLoc,
                dFbrPtr1 + dFbrLoc,
                dFbrIdx1 + dFbrIdxLoc,
                tile_nFibers,
                dU_dULoc2, // dU + dULoc[2],
                dU_dULoc0, // dU + dULoc[0],
                dU_dULoc1, // dU + dULoc[1],
                opt_mode, opt_R,
                warpPerFbr, logOfWarpPerFbr,
                item);
            });
          });

        else if (TiledX[0].ndims == 4) {
          q.submit([&] (sycl::handler &cgh) {
            auto dU_dULoc0 = dU + dULoc[0];
            auto dU_dULoc1 = dU + dULoc[1];
            auto dU_dULoc2 = dU + dULoc[2];
            auto dU_dULoc3 = dU + dULoc[3];
            cgh.parallel_for<class nnz_atomic_4d>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
              mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D(
                dVals + dLoc,
                dFbrLikeSlcInds + dFbrIdxLoc,
                dInds3 + dLoc,
                dFbrPtr0 + dSlcLoc,
                dFbrPtr1 + dFbrLoc,
                dFbrIdx1 + dFbrIdxLoc,
                dFbrPtr2 + dFbrLoc2,
                dFbrIdx2 + dFbrLoc2,
                tile_nFibers,
                dU_dULoc3,
                dU_dULoc0,
                dU_dULoc1,
                dU_dULoc2,
                opt_mode, opt_R,
                warpPerFbr, logOfWarpPerFbr,
                item);
            });
          });
	}
      }
      q.wait();
    }
  }

  /* Copying output matrix from GPU to CPU for correctness check */
  int MTTKRPmode = TiledX[0].ndims - 1;
  ITYPE loc = ((TiledX[0].ndims == 3) ? szDU[0] + szDU[1] : szDU[0] + szDU[1] + szDU[2]);

  q.memcpy(&U[MTTKRPmode].vals[0], dU + loc,
           U[MTTKRPmode].nRows * U[MTTKRPmode].nCols * sizeof(DTYPE)).wait();

  free(dVals, q);
  free(dU, q);
  free(dFbrIdx0, q);
  free(dFbrIdx1, q);
  free(dFbrPtr0, q);
  free(dFbrPtr1, q);
  free(dFbrLikeSlcInds, q);

  if(TiledX[0].ndims == 3)
    free(dInds2, q);

  if(TiledX[0].ndims == 4){
    free(dFbrIdx2, q);
    free(dFbrPtr2, q);
    free(dInds3, q);
  }

  delete[] dULoc;
  delete[] szDU;

  int totalMIslics = 0, totalMISfibers = 0, totalMIfibers = 0, totalMInnz = 0;;
  for (int m = 0; m <  TiledX[0].ndims; ++m){
    if(TiledX[m].totNnz){
      if(TiledX[m].ndims == 3){
        totalMIslics += TiledX[m].fbrIdx[0].size();
        totalMIfibers += TiledX[m].fbrPtr[1].size();
        totalMInnz += TiledX[m].totNnz;
      }

      if(TiledX[m].ndims == 4){
        totalMIslics += TiledX[m].fbrIdx[0].size();
        totalMISfibers += TiledX[m].fbrPtr[1].size();
        totalMIfibers += TiledX[m].fbrPtr[2].size();
        totalMInnz += TiledX[m].totNnz;
      }
    }
  }

  std::cout << "Resource usage: " << std::endl;
  if(TiledX[0].ndims == 3)
    std::cout << " nSlc:" << totalMIslics
      << ", nFibers:" << totalMIfibers << ", nnz:" << totalMInnz
      << std::endl;
  else if(TiledX[0].ndims == 4)
    std::cout << " nSlc:" << totalMIslics  << ", nSFibers:" << totalMISfibers
      << ", nFibers:" << totalMIfibers << ", nnz:" << totalMInnz
      << std::endl;

  return 0;
}
