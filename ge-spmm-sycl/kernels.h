#define __syncwarp() sg.barrier()

template<typename T>
void spmm_test1(
  sycl::nd_item<2> &item,
  int *__restrict colInd_sh,
  int A_nrows, int B_ncols,
  const int*__restrict A_csrRowPtr,
  const int*__restrict A_csrColInd,
  const T*__restrict A_csrVal,
  const T*__restrict B_dnVal,
        T*__restrict C_dnVal)
{
  const int blockDim_y = item.get_local_range(0);
  const int gridDim_y = item.get_group_range(0);
  const int threadIdx_y = item.get_local_id(0);
  const int blockIdx_y = item.get_group(0);
  const int blockIdx_x = item.get_group(1);
  const int threadIdx_x = item.get_local_id(1);
  auto sg = item.get_sub_group();

  T *val_sh = (T *)&colInd_sh[(blockDim_y<<5)];
  int shmem_offset = (threadIdx_y<<5);
  int thread_idx = shmem_offset+threadIdx_x;

  int rid = blockDim_y*blockIdx_x+threadIdx_y;

  if (rid<A_nrows) {
    int cid = (blockIdx_y<<5)+threadIdx_x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx_x;
    int offset;
    T acc=0;

    if (blockIdx_y != gridDim_y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
        }
        __syncwarp();
      }
      C_dnVal[(rid*B_ncols+cid)] = acc;
    }
    else {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (cid<B_ncols) {
            acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
          }
        }
        __syncwarp();
      }
      if (cid<B_ncols) {
        C_dnVal[(rid*B_ncols+cid)] = acc;
      }
    }
  }
}

template<typename T>
void spmm_test2(
  sycl::nd_item<2> &item,
  int *__restrict colInd_sh,
  int A_nrows, int B_ncols,
  const int*__restrict A_csrRowPtr,
  const int*__restrict A_csrColInd,
  const T*__restrict A_csrVal,
  const T*__restrict B_dnVal,
        T*__restrict C_dnVal)
{
  const int blockDim_y = item.get_local_range(0);
  const int gridDim_y = item.get_group_range(0);
  const int threadIdx_y = item.get_local_id(0);
  const int blockIdx_y = item.get_group(0);
  const int blockIdx_x = item.get_group(1);
  const int threadIdx_x = item.get_local_id(1);
  auto sg = item.get_sub_group();

  T *val_sh = (T *)&colInd_sh[(blockDim_y<<5)];
  int shmem_offset = (threadIdx_y<<5);
  int thread_idx = shmem_offset+threadIdx_x;

  int rid = blockDim_y*blockIdx_x+threadIdx_y;

  if (rid<A_nrows) {
    int cid = (blockIdx_y<<6)+threadIdx_x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx_x;
    int offset;
    T acc1=0, acc2=0, val;

    if (blockIdx_y != gridDim_y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          val = val_sh[(shmem_offset+kk)];
          acc1 += val*B_dnVal[offset];
          acc2 += val*B_dnVal[offset+32];
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      C_dnVal[offset] = acc1;
      C_dnVal[offset+32] = acc2;
    }
    else {
      int nout = (B_ncols-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          val = val_sh[(shmem_offset+kk)];
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (nout>0) {
            acc1 += val*B_dnVal[offset];
          }
          if (nout>1) {
            acc2 += val*B_dnVal[offset+32];  
          }
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      if (nout>0) {
        C_dnVal[offset] = acc1;
      }
      if (nout>1) {
        C_dnVal[(offset+32)] = acc2;
      }
    }
  }
}

template<typename T>
void spmm_test3(
  sycl::nd_item<2> &item,
  int *__restrict colInd_sh,
  int A_nrows, int B_ncols,
  const int*__restrict A_csrRowPtr,
  const int*__restrict A_csrColInd,
  const T*__restrict A_csrVal,
  const T*__restrict B_dnVal,
        T*__restrict C_dnVal)
{
  const int blockDim_y = item.get_local_range(0);
  const int gridDim_y = item.get_group_range(0);
  const int threadIdx_y = item.get_local_id(0);
  const int blockIdx_y = item.get_group(0);
  const int blockIdx_x = item.get_group(1);
  const int threadIdx_x = item.get_local_id(1);
  auto sg = item.get_sub_group();

  T *val_sh = (T *)&colInd_sh[(blockDim_y<<5)];
  int shmem_offset = (threadIdx_y<<5);
  int thread_idx = shmem_offset+threadIdx_x;

  int rid = blockDim_y*blockIdx_x+threadIdx_y;

  if (rid<A_nrows) {
    int cid = (blockIdx_y<<7)+threadIdx_x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx_x;
    int offset;
    T acc1=0, acc2=0, acc3=0, acc4=0, val;

    if (blockIdx_y != gridDim_y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          val = val_sh[(shmem_offset+kk)];
          acc1 += val*B_dnVal[offset];
          acc2 += val*B_dnVal[offset+32];
          acc3 += val*B_dnVal[offset+64];
          acc4 += val*B_dnVal[offset+96];
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      C_dnVal[offset] = acc1;
      C_dnVal[offset+32] = acc2;
      C_dnVal[offset+64] = acc3;
      C_dnVal[offset+96] = acc4;
    }
    else {
      int nout = (B_ncols-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          val = val_sh[(shmem_offset+kk)];
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (nout>0) {
            acc1 += val*B_dnVal[offset];
          }
          if (nout>1) {
            acc2 += val*B_dnVal[offset+32];  
          }
          if (nout>2) {
            acc3 += val*B_dnVal[offset+64];
          }
          if (nout>3) {
            acc4 += val*B_dnVal[offset+96];  
          }
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      if (nout>0) {
        C_dnVal[offset] = acc1;
      }
      if (nout>1) {
        C_dnVal[(offset+32)] = acc2;
      }
      if (nout>2) {
        C_dnVal[(offset+64)] = acc3;
      }
      if (nout>3) {
        C_dnVal[(offset+96)] = acc4;
      }
    }
  }
}

template<typename T>
void spmm_test4(
  sycl::nd_item<2> &item,
  int *__restrict colInd_sh,
  int A_nrows, int B_ncols,
  const int*__restrict A_csrRowPtr,
  const int*__restrict A_csrColInd,
  const T*__restrict A_csrVal,
  const T*__restrict B_dnVal,
        T*__restrict C_dnVal)
{
  const int blockDim_y = item.get_local_range(0);
  const int gridDim_y = item.get_group_range(0);
  const int threadIdx_y = item.get_local_id(0);
  const int blockIdx_y = item.get_group(0);
  const int blockIdx_x = item.get_group(1);
  const int threadIdx_x = item.get_local_id(1);
  auto sg = item.get_sub_group();

  T *val_sh = (T *)&colInd_sh[(blockDim_y<<5)];
  int shmem_offset = (threadIdx_y<<5);
  int thread_idx = shmem_offset+threadIdx_x;

  int rid = blockDim_y*blockIdx_x+threadIdx_y;

  if (rid<A_nrows) {
    int cid = (blockIdx_y<<8)+threadIdx_x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx_x;
    int offset;
    T acc1=0, acc2=0, acc3=0, acc4=0, acc5=0,acc6=0,acc7=0,acc8=0,val;

    if (blockIdx_y != gridDim_y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          val = val_sh[(shmem_offset+kk)];
          acc1 += val*B_dnVal[offset];
          acc2 += val*B_dnVal[offset+32];
          acc3 += val*B_dnVal[offset+64];
          acc4 += val*B_dnVal[offset+96];
          acc5 += val*B_dnVal[offset+128];
          acc6 += val*B_dnVal[offset+160];
          acc7 += val*B_dnVal[offset+192];
          acc8 += val*B_dnVal[offset+224];
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      C_dnVal[offset] = acc1;
      C_dnVal[offset+32] = acc2;
      C_dnVal[offset+64] = acc3;
      C_dnVal[offset+96] = acc4;
      C_dnVal[offset+128] = acc5;
      C_dnVal[offset+160] = acc6;
      C_dnVal[offset+192] = acc7;
      C_dnVal[offset+224] = acc8;
    }
    else {
      int nout = (B_ncols-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          val = val_sh[(shmem_offset+kk)];
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (nout>0) {
            acc1 += val*B_dnVal[offset];
          }
          if (nout>1) {
            acc2 += val*B_dnVal[offset+32];  
          }
          if (nout>2) {
            acc3 += val*B_dnVal[offset+64];
          }
          if (nout>3) {
            acc4 += val*B_dnVal[offset+96];  
          }
          if (nout>4) {
            acc5 += val*B_dnVal[offset+128];  
          }
          if (nout>5) {
            acc6 += val*B_dnVal[offset+160];  
          }
          if (nout>6) {
            acc7 += val*B_dnVal[offset+192];  
          }
          if (nout>7) {
            acc8 += val*B_dnVal[offset+224];  
          }
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      if (nout>0) {
        C_dnVal[offset] = acc1;
      }
      if (nout>1) {
        C_dnVal[(offset+32)] = acc2;
      }
      if (nout>2) {
        C_dnVal[(offset+64)] = acc3;
      }
      if (nout>3) {
        C_dnVal[(offset+96)] = acc4;
      }
      if (nout>4) {
        C_dnVal[(offset+128)] = acc5;
      }
      if (nout>5) {
        C_dnVal[(offset+160)] = acc6;
      }
      if (nout>6) {
        C_dnVal[(offset+192)] = acc7;
      }
      if (nout>7) {
        C_dnVal[(offset+224)] = acc8;
      }
    }
  }
}

