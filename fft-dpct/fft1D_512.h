#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void fft1D_512(T2 *work, sycl::nd_item<3> item_ct1, T *smem)
{

  int tid = item_ct1.get_local_id(2);
  int gid = item_ct1.get_group(2) * 512 + tid;
  int hi = tid>>3;
  int lo = tid&7;
  T2 data[8];
  //__local T smem[8*8*9];
  const int reversed[] = {0,4,2,6,1,5,3,7};

  // starting index of data to/from global memory
  // globalLoads8(data, work, 64)
  for( int i = 0; i < 8; i++ ) data[i] = work[gid+i*64];

  FFT8( data );

  //twiddle8( data, tid, 512 );
#ifdef UNROLL
  data[1] = cmplx_mul( data[1],exp_i(((T)-2*(T)M_PI*reversed[1]/(T)512)*tid) ); 
  data[2] = cmplx_mul( data[2],exp_i(((T)-2*(T)M_PI*reversed[2]/(T)512)*tid) ); 
  data[3] = cmplx_mul( data[3],exp_i(((T)-2*(T)M_PI*reversed[3]/(T)512)*tid) ); 
  data[4] = cmplx_mul( data[4],exp_i(((T)-2*(T)M_PI*reversed[4]/(T)512)*tid) ); 
  data[5] = cmplx_mul( data[5],exp_i(((T)-2*(T)M_PI*reversed[5]/(T)512)*tid) ); 
  data[6] = cmplx_mul( data[6],exp_i(((T)-2*(T)M_PI*reversed[6]/(T)512)*tid) ); 
  data[7] = cmplx_mul( data[7],exp_i(((T)-2*(T)M_PI*reversed[7]/(T)512)*tid) ); 
#else
  for( int j = 1; j < 8; j++ ){                                       
      data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)512)*tid) ); 
  }                                                                   
#endif

  //transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
    for (int i = 0; i < 8; i++)
        smem[hi * 8 + lo + i * 66] = data[reversed[i]].x();
  item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < 8; i++) data[i].x() = smem[lo * 66 + hi + i * 8];
  item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < 8; i++)
        smem[hi * 8 + lo + i * 66] = data[reversed[i]].y();
  item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < 8; i++) data[i].y() = smem[lo * 66 + hi + i * 8];
  item_ct1.barrier(sycl::access::fence_space::local_space);

  FFT8( data );

  //twiddle8( data, hi, 64 );
#ifdef UNROLL
  data[1] = cmplx_mul( data[1],exp_i(((T)-2*(T)M_PI*reversed[1]/(T)64)*hi) ); 
  data[2] = cmplx_mul( data[2],exp_i(((T)-2*(T)M_PI*reversed[2]/(T)64)*hi) ); 
  data[3] = cmplx_mul( data[3],exp_i(((T)-2*(T)M_PI*reversed[3]/(T)64)*hi) ); 
  data[4] = cmplx_mul( data[4],exp_i(((T)-2*(T)M_PI*reversed[4]/(T)64)*hi) ); 
  data[5] = cmplx_mul( data[5],exp_i(((T)-2*(T)M_PI*reversed[5]/(T)64)*hi) ); 
  data[6] = cmplx_mul( data[6],exp_i(((T)-2*(T)M_PI*reversed[6]/(T)64)*hi) ); 
  data[7] = cmplx_mul( data[7],exp_i(((T)-2*(T)M_PI*reversed[7]/(T)64)*hi) ); 
#else
  for( int j = 1; j < 8; j++ ){                                       
      data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)64)*hi) ); 
  }                                                                   
#endif

  //transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
    for (int i = 0; i < 8; i++)
        smem[hi * 8 + lo + i * 72] = data[reversed[i]].x();
  item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < 8; i++) data[i].x() = smem[hi * 72 + lo + i * 8];
  item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < 8; i++)
        smem[hi * 8 + lo + i * 72] = data[reversed[i]].y();
  item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < 8; i++) data[i].y() = smem[hi * 72 + lo + i * 8];

  FFT8( data );

  //globalStores8(data, work, 64);
  for( int i = 0; i < 8; i++ )
    work[gid+i*64] = data[reversed[i]];
}
