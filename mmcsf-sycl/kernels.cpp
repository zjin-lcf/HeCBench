#include <iostream>
#include "mttkrp_gpu.h"
#include <vector>

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
    ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
    ITYPE  mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int fbrPerWarp, int logOfFPW, 
    nd_item<1> &item){

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
          //atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
          auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                       ext::oneapi::memory_order::relaxed, 
                       ext::oneapi::memory_scope::device, 
                       access::address_space::global_space> (dU0[idx0 * R + r]);
          atomic_obj_ref.fetch_add(tmp);
        } 
        tmp = 0;
      }
    } 

    if(!diffFiber) {  
      for(unsigned int r=laneId; r<R; r+=32) { 
        //atomicAdd(&dU0[idx0 * R + r], tmp); 
        auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                     ext::oneapi::memory_order::relaxed, 
                     ext::oneapi::memory_scope::device, 
                     access::address_space::global_space> (dU0[idx0 * R + r]);
        atomic_obj_ref.fetch_add(tmp);
      }  
    }  
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds3, 
    ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, ITYPE nFibers, DTYPE *dU0, 
    DTYPE * dU1, DTYPE *dU2, DTYPE *dU3, ITYPE  mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int fbrPerWarp, 
    int logOfFPW, nd_item<1> &item){

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
          //atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR
          auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                       ext::oneapi::memory_order::relaxed, 
                       ext::oneapi::memory_scope::device, 
                       access::address_space::global_space> (dU0[idx0 * R + r]);
          atomic_obj_ref.fetch_add(tmp2);
        } 
        tmp2 = 0;
      }
    }

    if(!diffFiber) {  
      for(unsigned int r=laneId; r<R; r+=32) {
        //atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR           
        auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                     ext::oneapi::memory_order::relaxed, 
                     ext::oneapi::memory_scope::device, 
                     access::address_space::global_space> (dU0[idx0 * R + r]);
        atomic_obj_ref.fetch_add(tmp2);
      }
    }
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D(DTYPE * vals, ITYPE *fbrLikeSlcInds,  ITYPE *dInds3, 
    ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, ITYPE nFibers, DTYPE *dU0,
    DTYPE * dU1, DTYPE *dU2, DTYPE *dU3, ITYPE  mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, 
    nd_item<1> &item){

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
      //atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR
      auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                   ext::oneapi::memory_order::relaxed, 
                   ext::oneapi::memory_scope::device, 
                   access::address_space::global_space> (dU0[idx0 * R + r]);
      atomic_obj_ref.fetch_add(tmp2);
    }    
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
    ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
    ITYPE  mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC,
    nd_item<1> &item){

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
      //atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
      auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                   ext::oneapi::memory_order::relaxed, 
                   ext::oneapi::memory_scope::device, 
                   access::address_space::global_space> (dU0[idx0 * R + r]);
      atomic_obj_ref.fetch_add(tmp);

    }    
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D(DTYPE * vals, ITYPE *fbrLikeSlcInds,  ITYPE *dInds3, 
    ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, ITYPE nFibers, DTYPE *dU0,
    DTYPE * dU1, DTYPE *dU2, DTYPE *dU3, ITYPE  mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC,
    nd_item<1> &item){

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
        //atomicAdd(&dU0[idx0 * R + r], tmp * dU2[idx2 * R + r] * dU3[idx3 * R + r]) ;  
        auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                     ext::oneapi::memory_order::relaxed, 
                     ext::oneapi::memory_scope::device, 
                     access::address_space::global_space> (dU0[idx0 * R + r]);
        atomic_obj_ref.fetch_add(tmp * dU2[idx2 * R + r] * dU3[idx3 * R + r]);
      }
    }            
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar(DTYPE * vals, ITYPE *fbrLikeSlcInds, ITYPE *dInds2, 
    ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE nFibers, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
    ITYPE  mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC,
    nd_item<1> &item){

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
        //atomicAdd(&dU0[idx0 * R + r], tmp_val);
        auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                     ext::oneapi::memory_order::relaxed, 
                     ext::oneapi::memory_scope::device, 
                     access::address_space::global_space> (dU0[idx0 * R + r]);
        atomic_obj_ref.fetch_add(tmp_val);
      }
    }         
  }
}

// fbr atomic sing slcLikeFbr
void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D(DTYPE * vals, ITYPE *fbrLikeSlcInds,  ITYPE *dInds3, 
    ITYPE *fbrPtr0, ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, ITYPE nFibers, DTYPE *dU0,
    DTYPE * dU1, DTYPE *dU2, DTYPE *dU3, ITYPE  mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC,
    nd_item<1> &item){

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
          //atomicAdd(&dU0[idx0 * R + r], tmp);
          auto atomic_obj_ref = ext::oneapi::atomic_ref<DTYPE,
                       ext::oneapi::memory_order::relaxed, 
                       ext::oneapi::memory_scope::device, 
                       access::address_space::global_space> (dU0[idx0 * R + r]);
          atomic_obj_ref.fetch_add(tmp);
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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  sycl::queue q(dev_sel);

  // Allocate Tensor on a device
  buffer<DTYPE, 1> dVals (totNnz);
  buffer<ITYPE, 1> dFbrPtr0 (totSlcPtr);
  buffer<ITYPE, 1> dFbrIdx0 (totSlcIdx);
  buffer<ITYPE, 1> dFbrPtr1 (totFbrPtr);
  buffer<ITYPE, 1> dFbrIdx1 (totFbrIdx);
  buffer<ITYPE, 1> dFbrLikeSlcInds (totFbrIdx);

  // conditional buffer allocation will produce undefined variables
  // ndim = 3
  buffer<ITYPE, 1> dInds2 (totNnz);
  // ndim = 4
  buffer<ITYPE, 1> dFbrIdx2 (totFbrPtr2);
  buffer<ITYPE, 1> dFbrPtr2 (totFbrPtr2);
  buffer<ITYPE, 1> dInds3 (totNnz);

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

    q.submit([&] (handler &cgh) {
      auto acc = dVals.get_access<sycl_write>(cgh, range<1>(TiledX[m].totNnz), id<1>(dLoc));
      cgh.copy(&(TiledX[m].vals[0]), acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = dFbrPtr0.get_access<sycl_write>(cgh, range<1>(TiledX[m].fbrPtr[0].size()), id<1>(dSlcLoc));
      cgh.copy(&(TiledX[m].fbrPtr[0][0]), acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = dFbrIdx0.get_access<sycl_write>(cgh, range<1>(TiledX[m].fbrIdx[0].size()), id<1>(dSlcIdxLoc));
      cgh.copy(&(TiledX[m].fbrIdx[0][0]), acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = dFbrPtr1.get_access<sycl_write>(cgh, range<1>(TiledX[m].fbrPtr[1].size()), id<1>(dFbrLoc));
      cgh.copy(&(TiledX[m].fbrPtr[1][0]), acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = dFbrIdx1.get_access<sycl_write>(cgh, range<1>(TiledX[m].fbrIdx[1].size()), id<1>(dFbrIdxLoc));
      cgh.copy(&(TiledX[m].fbrIdx[1][0]), acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = dFbrLikeSlcInds.get_access<sycl_write>(cgh, range<1>(TiledX[m].fbrIdx[1].size()), id<1>(dFbrIdxLoc));
      cgh.copy(&(TiledX[m].fbrLikeSlcInds[0]), acc);
    });

    if(TiledX[m].ndims == 3){
      if(m <= 2)
        q.submit([&] (handler &cgh) {
          auto acc = dInds2.get_access<sycl_write>(cgh, range<1>(TiledX[m].totNnz), id<1>(dLoc));
          cgh.copy(&(TiledX[m].inds[TiledX[m].modeOrder[2]][0]), acc);
        });
    }
    if(TiledX[m].ndims == 4){      
        q.submit([&] (handler &cgh) {
          auto acc = dFbrPtr2.get_access<sycl_write>(cgh, range<1>(TiledX[m].fbrPtr[2].size()), id<1>(dFbrLoc2));
          cgh.copy(&(TiledX[m].fbrPtr[2][0]), acc);
        });

        q.submit([&] (handler &cgh) {
          auto acc = dFbrIdx2.get_access<sycl_write>(cgh, range<1>(TiledX[m].fbrIdx[2].size()), id<1>(dFbrLoc2));
          cgh.copy(&(TiledX[m].fbrIdx[2][0]), acc);
        });

        q.submit([&] (handler &cgh) {
          auto acc = dInds3.get_access<sycl_write>(cgh, range<1>(TiledX[m].totNnz), id<1>(dLoc));
          cgh.copy(&(TiledX[m].inds[TiledX[m].modeOrder[3]][0]), acc);
        });
    }
  }

  //Matrices
  unsigned int *dULoc =  new unsigned int[TiledX[0].ndims];
  unsigned int *szDU =  new unsigned int[TiledX[0].ndims];

  //Matrices
  ITYPE mtxSize = ((TiledX[0].ndims == 3) ? (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows) * U[mode0].nCols
      : (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows + U[mode3].nRows) * U[mode0].nCols );

  buffer<DTYPE, 1> dU (mtxSize); 

  for (int m = 0; m < TiledX[0].ndims; ++m)
    szDU[m] = U[m].nRows * U[m].nCols;

  q.submit([&] (handler &cgh) {
    auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode0].nRows * U[mode0].nCols));
    cgh.fill(acc, (DTYPE)0);
  });

  q.submit([&] (handler &cgh) {
    auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode1].nRows * U[mode1].nCols), id<1>(szDU[0]));
    cgh.copy(&(U[mode1].vals[0]), acc);
  });

  q.submit([&] (handler &cgh) {
    auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode2].nRows * U[mode2].nCols), id<1>(szDU[0]+szDU[1]));
    cgh.copy(&(U[mode2].vals[0]), acc);
  });

  if(TiledX[0].ndims == 4)
    q.submit([&] (handler &cgh) {
      auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode3].nRows * U[mode3].nCols), 
                                           id<1>(szDU[0]+szDU[1]+szDU[2]));
      cgh.copy(&(U[mode3].vals[0]), acc);
    });


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

        q.submit([&] (handler &cgh) {
          auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode0].nRows * U[mode0].nCols));
          cgh.copy(&(U[mode0].vals[0]), acc);
        });

        q.submit([&] (handler &cgh) {
          auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode1].nRows * U[mode1].nCols), id<1>(szDU[0]));
          cgh.fill(acc, (DTYPE)0);
        });

      }
      else if(MTTKRPmode == 2){
        q.submit([&] (handler &cgh) {
          auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode1].nRows * U[mode1].nCols), id<1>(szDU[0]));
          cgh.copy(&(U[mode1].vals[0]), acc);
        });

        q.submit([&] (handler &cgh) {
          auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode2].nRows * U[mode2].nCols), 
                                               id<1>(szDU[0]+szDU[1]));
          cgh.fill(acc, (DTYPE)0);
        });
      }
      else if(MTTKRPmode == 3){
        q.submit([&] (handler &cgh) {
          auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode2].nRows * U[mode2].nCols), 
                                               id<1>(szDU[0]+szDU[1]));
          cgh.copy(&(U[mode2].vals[0]), acc);
        });

        q.submit([&] (handler &cgh) {
          auto acc = dU.get_access<sycl_write>(cgh, range<1>(U[mode3].nRows * U[mode3].nCols), 
                                               id<1>(szDU[0]+szDU[1]+szDU[2]));
          cgh.fill(acc, (DTYPE)0);
        });
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
        range<1> gws (grid * BLOCKSIZE);
        range<1> lws (BLOCKSIZE);

        if(TiledX[0].ndims == 3)
          q.submit([&] (handler &cgh) {
            auto dVals_acc = dVals.get_access<sycl_read>(cgh);
            auto dFbrLikeSlcInds_acc = dFbrLikeSlcInds.get_access<sycl_read>(cgh);
            auto dInds2_acc = dInds2.get_access<sycl_read>(cgh);
            auto dFbrPtr0_acc = dFbrPtr0.get_access<sycl_read>(cgh);
            auto dFbrPtr1_acc = dFbrPtr1.get_access<sycl_read>(cgh);
            auto dFbrIdx1_acc = dFbrIdx1.get_access<sycl_read>(cgh);
            auto dU_acc = dU.get_access<sycl_read_write>(cgh);
            cgh.parallel_for<class slc_atomic>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
              mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar(
                dVals_acc.get_pointer() + dLoc, 
                dFbrLikeSlcInds_acc.get_pointer() + dFbrIdxLoc, 
                dInds2_acc.get_pointer() + dLoc, 
                dFbrPtr0_acc.get_pointer() + dSlcLoc, 
                dFbrPtr1_acc.get_pointer() + dFbrLoc,
                dFbrIdx1_acc.get_pointer() + dFbrIdxLoc,
                tile_nFibers, 
                dU_acc.get_pointer() + dULoc[0], 
                dU_acc.get_pointer() + dULoc[1],
                dU_acc.get_pointer() + dULoc[2],
                opt_mode, opt_R,
                warpPerFbr, logOfWarpPerFbr, 
                fbrPerWarp, logOfFbrPerWarp, 
                item);
            });
          });

        else if(TiledX[0].ndims == 4) {
          q.submit([&] (handler &cgh) {
            auto dVals_acc = dVals.get_access<sycl_read>(cgh);
            auto dFbrLikeSlcInds_acc = dFbrLikeSlcInds.get_access<sycl_read>(cgh);
            auto dInds3_acc = dInds3.get_access<sycl_read>(cgh);
            auto dFbrPtr0_acc = dFbrPtr0.get_access<sycl_read>(cgh);
            auto dFbrPtr1_acc = dFbrPtr1.get_access<sycl_read>(cgh);
            auto dFbrPtr2_acc = dFbrPtr2.get_access<sycl_read>(cgh);
            auto dFbrIdx1_acc = dFbrIdx1.get_access<sycl_read>(cgh);
            auto dFbrIdx2_acc = dFbrIdx2.get_access<sycl_read>(cgh);
            auto dU_acc = dU.get_access<sycl_read_write>(cgh);
            cgh.parallel_for<class slc_atomic_4d>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
              mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D(
                dVals_acc.get_pointer() + dLoc, 
                dFbrLikeSlcInds_acc.get_pointer() + dFbrIdxLoc, 
                dInds3_acc.get_pointer() + dLoc, 
                dFbrPtr0_acc.get_pointer() + dSlcLoc,
                dFbrPtr1_acc.get_pointer() + dFbrLoc,
                dFbrIdx1_acc.get_pointer() + dFbrIdxLoc,
                dFbrPtr2_acc.get_pointer() + dFbrLoc2,
                dFbrIdx2_acc.get_pointer() + dFbrLoc2, 
                tile_nFibers, 
                dU_acc.get_pointer() + dULoc[0], 
                dU_acc.get_pointer() + dULoc[1], 
                dU_acc.get_pointer() + dULoc[2], 
                dU_acc.get_pointer() + dULoc[3], 
                opt_mode, opt_R, 
                warpPerFbr, logOfWarpPerFbr, 
                fbrPerWarp, logOfFbrPerWarp,
                item);
            });
          });
	  q.wait();
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
        range<1> gws (grid * BLOCKSIZE);
        range<1> lws (BLOCKSIZE);

        q.submit([&] (handler &cgh) {
          auto dVals_acc = dVals.get_access<sycl_read>(cgh);
          auto dFbrLikeSlcInds_acc = dFbrLikeSlcInds.get_access<sycl_read>(cgh);
          auto dInds3_acc = dInds3.get_access<sycl_read>(cgh);
          auto dFbrPtr0_acc = dFbrPtr0.get_access<sycl_read>(cgh);
          auto dFbrPtr1_acc = dFbrPtr1.get_access<sycl_read>(cgh);
          auto dFbrPtr2_acc = dFbrPtr2.get_access<sycl_read>(cgh);
          auto dFbrIdx1_acc = dFbrIdx1.get_access<sycl_read>(cgh);
          auto dFbrIdx2_acc = dFbrIdx2.get_access<sycl_read>(cgh);
          auto dU_acc = dU.get_access<sycl_read_write>(cgh);
          cgh.parallel_for<class fbrs_atomic>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
            mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D(
              dVals_acc.get_pointer() + dLoc, 
              dFbrLikeSlcInds_acc.get_pointer() + dFbrIdxLoc, 
              dInds3_acc.get_pointer() + dLoc, 
              dFbrPtr0_acc.get_pointer() + dSlcLoc, 
              dFbrPtr1_acc.get_pointer() + dFbrLoc,
              dFbrIdx1_acc.get_pointer() + dFbrIdxLoc,
              dFbrPtr2_acc.get_pointer() + dFbrLoc2,
              dFbrIdx2_acc.get_pointer() + dFbrLoc2,
              tile_nFibers, 
              dU_acc.get_pointer() + dULoc[1],
              dU_acc.get_pointer() + dULoc[2],
              dU_acc.get_pointer() + dULoc[3],
              dU_acc.get_pointer() + dULoc[0], 
              opt_mode, opt_R, 
              warpPerFbr, logOfWarpPerFbr, 
              item);
          });
        });
	q.wait();
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
        range<1> gws (grid * BLOCKSIZE);
        range<1> lws (BLOCKSIZE);

        if(TiledX[0].ndims == 3)
          q.submit([&] (handler &cgh) {
            auto dVals_acc = dVals.get_access<sycl_read>(cgh);
            auto dFbrLikeSlcInds_acc = dFbrLikeSlcInds.get_access<sycl_read>(cgh);
            auto dInds2_acc = dInds2.get_access<sycl_read>(cgh);
            auto dFbrPtr0_acc = dFbrPtr0.get_access<sycl_read>(cgh);
            auto dFbrPtr1_acc = dFbrPtr1.get_access<sycl_read>(cgh);
            auto dFbrIdx1_acc = dFbrIdx1.get_access<sycl_read>(cgh);
            auto dU_acc = dU.get_access<sycl_read_write>(cgh);
            cgh.parallel_for<class fbr_atomic>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
              mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar(
                dVals_acc.get_pointer() + dLoc, 
                dFbrLikeSlcInds_acc.get_pointer() + dFbrIdxLoc, 
                dInds2_acc.get_pointer() + dLoc, 
                dFbrPtr0_acc.get_pointer() + dSlcLoc, 
                dFbrPtr1_acc.get_pointer() + dFbrLoc,  
                dFbrIdx1_acc.get_pointer() + dFbrIdxLoc, 
                tile_nFibers, 
                dU_acc.get_pointer() + dULoc[1], 
                dU_acc.get_pointer() + dULoc[2],
                dU_acc.get_pointer() + dULoc[0],
                opt_mode, opt_R, 
                warpPerFbr, logOfWarpPerFbr,
                item);
            });
          });

        else if (TiledX[0].ndims == 4) {
          q.submit([&] (handler &cgh) {
            auto dVals_acc = dVals.get_access<sycl_read>(cgh);
            auto dFbrLikeSlcInds_acc = dFbrLikeSlcInds.get_access<sycl_read>(cgh);
            auto dInds3_acc = dInds3.get_access<sycl_read>(cgh);
            auto dFbrPtr0_acc = dFbrPtr0.get_access<sycl_read>(cgh);
            auto dFbrPtr1_acc = dFbrPtr1.get_access<sycl_read>(cgh);
            auto dFbrIdx1_acc = dFbrIdx1.get_access<sycl_read>(cgh);
            auto dFbrPtr2_acc = dFbrPtr2.get_access<sycl_read>(cgh);
            auto dFbrIdx2_acc = dFbrIdx2.get_access<sycl_read>(cgh);
            auto dU_acc = dU.get_access<sycl_read_write>(cgh);
            cgh.parallel_for<class fbr_atomic_4d>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
              mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D(
                dVals_acc.get_pointer() + dLoc,
                dFbrLikeSlcInds_acc.get_pointer() + dFbrIdxLoc, 
                dInds3_acc.get_pointer() + dLoc, 
                dFbrPtr0_acc.get_pointer() + dSlcLoc,
                dFbrPtr1_acc.get_pointer() + dFbrLoc,
                dFbrIdx1_acc.get_pointer() + dFbrIdxLoc,
                dFbrPtr2_acc.get_pointer() + dFbrLoc2,
                dFbrIdx2_acc.get_pointer() + dFbrLoc2, 
                tile_nFibers,
                dU_acc.get_pointer() + dULoc[2],
                dU_acc.get_pointer() + dULoc[3],
                dU_acc.get_pointer() + dULoc[0],
                dU_acc.get_pointer() + dULoc[1],
                opt_mode, opt_R, 
                warpPerFbr, logOfWarpPerFbr,
                item);
            });
          });
	  q.wait();
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
        range<1> gws (grid * BLOCKSIZE);
        range<1> lws (BLOCKSIZE);

        if (TiledX[0].ndims == 3)
          q.submit([&] (handler &cgh) {
            auto dVals_acc = dVals.get_access<sycl_read>(cgh);
            auto dFbrLikeSlcInds_acc = dFbrLikeSlcInds.get_access<sycl_read>(cgh);
            auto dInds2_acc = dInds2.get_access<sycl_read>(cgh);
            auto dFbrPtr0_acc = dFbrPtr0.get_access<sycl_read>(cgh);
            auto dFbrPtr1_acc = dFbrPtr1.get_access<sycl_read>(cgh);
            auto dFbrIdx1_acc = dFbrIdx1.get_access<sycl_read>(cgh);
            auto dU_acc = dU.get_access<sycl_read_write>(cgh);
            cgh.parallel_for<class nnz_atomic>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
              mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar(
                dVals_acc.get_pointer() + dLoc, 
                dFbrLikeSlcInds_acc.get_pointer() + dFbrIdxLoc, 
                dInds2_acc.get_pointer() + dLoc,
                dFbrPtr0_acc.get_pointer() + dSlcLoc,
                dFbrPtr1_acc.get_pointer() + dFbrLoc,
                dFbrIdx1_acc.get_pointer() + dFbrIdxLoc,
                tile_nFibers, 
                dU_acc.get_pointer() + dULoc[2],
                dU_acc.get_pointer() + dULoc[0], 
                dU_acc.get_pointer() + dULoc[1], 
                opt_mode, opt_R, 
                warpPerFbr, logOfWarpPerFbr,
                item); 
            });
          });

        else if (TiledX[0].ndims == 4) {
          q.submit([&] (handler &cgh) {
            auto dVals_acc = dVals.get_access<sycl_read>(cgh);
            auto dFbrLikeSlcInds_acc = dFbrLikeSlcInds.get_access<sycl_read>(cgh);
            auto dInds3_acc = dInds3.get_access<sycl_read>(cgh);
            auto dFbrPtr0_acc = dFbrPtr0.get_access<sycl_read>(cgh);
            auto dFbrPtr1_acc = dFbrPtr1.get_access<sycl_read>(cgh);
            auto dFbrIdx1_acc = dFbrIdx1.get_access<sycl_read>(cgh);
            auto dFbrPtr2_acc = dFbrPtr2.get_access<sycl_read>(cgh);
            auto dFbrIdx2_acc = dFbrIdx2.get_access<sycl_read>(cgh);
            auto dU_acc = dU.get_access<sycl_read_write>(cgh);
            cgh.parallel_for<class nnz_atomic_4d>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
              mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D(
                dVals_acc.get_pointer() + dLoc, 
                dFbrLikeSlcInds_acc.get_pointer() + dFbrIdxLoc, 
                dInds3_acc.get_pointer() + dLoc,
                dFbrPtr0_acc.get_pointer() + dSlcLoc,
                dFbrPtr1_acc.get_pointer() + dFbrLoc,
                dFbrIdx1_acc.get_pointer() + dFbrIdxLoc,
                dFbrPtr2_acc.get_pointer() + dFbrLoc2,
                dFbrIdx2_acc.get_pointer() + dFbrLoc2, 
                tile_nFibers,
                dU_acc.get_pointer() + dULoc[3],
                dU_acc.get_pointer() + dULoc[0],
                dU_acc.get_pointer() + dULoc[1],
                dU_acc.get_pointer() + dULoc[2],
                opt_mode, opt_R, 
                warpPerFbr, logOfWarpPerFbr,
                item);
            });
          });
	  q.wait();
	}
      }
    }
  }

  /* Copying output matrix from GPU to CPU for correctness check */
  int mttkrp_mode = TiledX[0].ndims - 1;
  ITYPE loc = ((TiledX[0].ndims == 3) ? szDU[0] + szDU[1] : szDU[0] + szDU[1] + szDU[2]);

  q.submit([&] (handler &cgh) {
    auto dU_acc = dU.get_access<sycl_read>(cgh, range<1>( U[mttkrp_mode].nRows * U[mttkrp_mode].nCols), id<1>(loc));
    cgh.copy(dU_acc, &U[mttkrp_mode].vals[0]);
  });

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

  q.wait();
  return 0;
}

