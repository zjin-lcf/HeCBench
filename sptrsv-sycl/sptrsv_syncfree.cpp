#ifndef _SPTRSV_SYNCFREE_
#define _SPTRSV_SYNCFREE_

#include "common.h"
#include "sptrsv.h"

int atomicLoad(nd_item<1> item, const int *addr)
{
  const volatile int *vaddr = addr; // volatile to bypass cache
  //__threadfence(); // for seq_cst loads. Remove for acquire semantics.
  const int value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  item.mem_fence(access::fence_space::global_space);
  return value; 
}

// addr must be aligned properly.
void atomicStore(nd_item<1> item, int *addr, int value)
{
  volatile int *vaddr = addr; // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other threads
  item.mem_fence(access::fence_space::global_space);
  *vaddr = value;
}


int sptrsv_syncfree (
    const int           *csrRowPtr,
    const int           *csrColIdx,
    const VALUE_TYPE    *csrVal,
    const int            m,
    const int            n,
    const int            nnz,
    VALUE_TYPE    *x,
    const VALUE_TYPE    *b,
    const VALUE_TYPE    *x_ref)
{
  if (m != n)
  {
    printf("This is not a square matrix, return.\n");
    return -1;
  }

  int *warp_num=(int *)malloc((m+1)*sizeof(int));
  memset (warp_num, 0, sizeof(int)*(m+1));

  memset(x, 0, m * sizeof(VALUE_TYPE));

  int *get_value = (int *)malloc(m * sizeof(int));
  memset(get_value, 0, m * sizeof(int));

  double warp_occupy=0,element_occupy=0;
  int Len;

  for(int i=0;i<BENCH_REPEAT;i++)
  {
    matrix_warp4(m,n,nnz,csrRowPtr,csrColIdx,csrVal,10,&Len,warp_num,&warp_occupy,&element_occupy);
  }


#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // Matrix L
  buffer<int, 1> d_csrRowPtr(csrRowPtr, m+1);
  buffer<int, 1> d_csrColIdx(csrColIdx, nnz);
  buffer<VALUE_TYPE, 1> d_csrVal(csrVal, nnz);

  // Vector b
  buffer<VALUE_TYPE, 1> d_b (b, m);

  // Vector x
  buffer<VALUE_TYPE, 1> d_x (n);

  buffer<int, 1> d_get_value (m);
  buffer<int, 1> d_warp_num (warp_num, Len);

  int num_threads = WARP_PER_BLOCK * WARP_SIZE;
  int num_blocks = ceil ((double)((Len-1)*WARP_SIZE) / (double)(num_threads));

  range<1> gws (num_blocks * num_threads);
  range<1> lws (num_threads);

  for (int i = 0; i < BENCH_REPEAT; i++)
  {

    // memset d_get_value to 0
    q.submit([&] (handler &cgh) {
      auto acc = d_get_value.get_access<sycl_write>(cgh);
      cgh.copy(get_value, acc);
    });
    q.submit([&] (handler &cgh) {
      auto acc = d_x.get_access<sycl_write>(cgh);
      cgh.copy(x, acc);
    });

    q.submit([&] (handler &cgh) {
      auto csrRowPtr = d_csrRowPtr.get_access<sycl_read>(cgh);
      auto csrColIdx = d_csrColIdx.get_access<sycl_read>(cgh);
      auto csrVal = d_csrVal.get_access<sycl_read>(cgh);
      auto b = d_b.get_access<sycl_read>(cgh);
      auto warp_num = d_warp_num.get_access<sycl_read>(cgh);
#ifdef FROM_OPENCL
      auto get_value = d_get_value.get_access<sycl_atomic>(cgh);
#else
      auto get_value = d_get_value.get_access<sycl_read_write>(cgh);
#endif
      auto x = d_x.get_access<sycl_read_write>(cgh);

      accessor<VALUE_TYPE, 1, sycl_read_write, access::target::local> 
        s_left_sum(WARP_PER_BLOCK*WARP_SIZE, cgh);

      cgh.parallel_for<class sptrsv_mix>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {

        const int global_id = item.get_global_id(0);
        const int warp_id = global_id/WARP_SIZE;
        const int local_id = item.get_local_id(0);

        int row;

        if(warp_id>=(Len-1)) return;

        const int lane_id = (WARP_SIZE - 1) & local_id;

        if(warp_num[warp_id+1]>(warp_num[warp_id]+1))
        {
          //thread
          row =warp_num[warp_id]+lane_id;
          if(row>=m) return;

          int col,j,i;
          VALUE_TYPE xi;
          VALUE_TYPE left_sum=0;
          i=row;
          j=csrRowPtr[i];

          while(j<csrRowPtr[i+1])
          {
            col=csrColIdx[j];
#ifdef FROM_OPENCL
            if(atomic_load(get_value[col])==1)
#else
            if(atomicLoad(item, &get_value[col])==1)
#endif
            {
              left_sum+=csrVal[j]*x[col];
              j++;
              col=csrColIdx[j];
            }
            if(i==col)
            {
              xi = (b[i] - left_sum) / csrVal[csrRowPtr[i+1]-1];
              x[i] = xi;
#ifdef FROM_OPENCL
              item.mem_fence(access::fence_space::global_space);
              get_value[i].store(1);
#else
              atomicStore(item, &get_value[i], 1);
#endif
              j++;
            }
          }
        }
        else
        {
          row = warp_num[warp_id];
          if(row>=m)
            return;

          int col,j=csrRowPtr[row]  + lane_id;
          VALUE_TYPE xi,sum=0;
          while(j < (csrRowPtr[row+1]-1))
          {
            col=csrColIdx[j];
#ifdef FROM_OPENCL
            if(atomic_load(get_value[col])==1)
#else
            if(atomicLoad(item, &get_value[col])==1)
#endif
            {
              sum += x[col] * csrVal[j];
              j += WARP_SIZE;
            }
          }

          s_left_sum[local_id]=sum;

          for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
          {
            if(lane_id < offset)
            {
              s_left_sum[local_id] += s_left_sum[local_id+offset];
            }
          }

          if (!lane_id)
          {
            xi = (b[row] - s_left_sum[local_id]) / csrVal[csrRowPtr[row+1]-1];
            x[row]=xi;
#ifdef FROM_OPENCL
            item.mem_fence(access::fence_space::global_space);
            get_value[row].store(1);
#else
            atomicStore(item, &get_value[i], 1);
#endif
          }
        }

      });
    });
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_x.get_access<sycl_read>(cgh);
    cgh.copy(acc, x);
  });

  q.wait();


  // validate x
  double accuracy = 1e-4;
  double ref = 0.0;
  double res = 0.0;
  int error=0;
  const int rhs = 1;

  for (int i = 0; i < n * rhs; i++)
  {
    ref += abs(x_ref[i]);
    res += abs(x[i] - x_ref[i]);
    if(x[i] != x_ref[i]  && error<10)
    {
      // printf("%d %f %f\n",i,x[i],x_ref[i]);
      error++;
    }
  }
  res = ref == 0 ? res : res / ref;

  if (res < accuracy)
    printf("syncfree SpTRSV passed! |x-xref|/|xref| = %8.2e\n", res);
  else
    printf("syncfree SpTRSV failed! |x-xref|/|xref| = %8.2e\n", res);

  free(get_value);
  free(warp_num);

  return 0;
}

#endif



