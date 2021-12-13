#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define  bidx  item.get_group(0)
#define  tidx  item.get_local_id(0)

#include "modP.h"

void intt_3_64k_modcrt(
        nd_item<1> &item,
        uint64 *__restrict buffer,
        uint32 *__restrict dst,
  const uint64 *__restrict src)
{
  register uint64 samples[8], s8[8];
  register uint32 fmem, tmem, fbuf, tbuf;
  fmem = (bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
  tbuf = tidx<<3;
  fbuf = ((tidx&0x38)<<3) | (tidx&0x7);
  tmem = (bidx<<9)|((tidx&0x38)<<3) | (tidx&0x7);
#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = src[fmem|(i<<1)];
  ntt8(samples);

#pragma unroll
  for (int i=0; i<8; i++)
    buffer[tbuf|i] = _ls_modP(samples[i], ((tidx&0x1)<<2)*i*3);
  item.barrier(access::fence_space::local_space);

#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|(i<<3)];

#pragma unroll
  for (int i=0; i<4; i++) {
    s8[2*i] = _add_modP(samples[2*i], samples[2*i+1]);
    s8[2*i+1] = _sub_modP(samples[2*i], samples[2*i+1]);
  }

#pragma unroll
  for (int i=0; i<8; i++) {
    dst[(((tmem|(i<<3))&0xf)<<12)|((tmem|(i<<3))>>4)] =
      (uint32)(_mul_modP(s8[i], 18446462594437939201UL, valP));
  }
}

int main() {
  const int nttLen = 64 * 1024;
  uint64 *ntt = (uint64*) malloc (nttLen*sizeof(uint64));
  uint32 *res = (uint32*) malloc (nttLen*sizeof(uint32));

  srand(123);
  for (int i = 0; i < nttLen; i++) {
    uint64 hi = rand();
    uint64 lo = rand();
    ntt[i] = (hi << 32) | lo;
  }

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<uint64, 1> d_ntt (ntt, nttLen);
  buffer<uint32, 1> d_res (res, nttLen);

  range<1> gws (nttLen/512 * 64);
  range<1> lws (64);
  
  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto dst = d_res.get_access<sycl_discard_write>(cgh);
      auto src = d_ntt.get_access<sycl_read>(cgh);
      accessor<uint64, 1, sycl_read_write, access::target::local> sm (512, cgh);
      cgh.parallel_for<class intt_modcrt>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        intt_3_64k_modcrt(item, sm.get_pointer(), dst.get_pointer(), src.get_pointer());
      });
    });
  }

  uint64_t checksum = 0;
  for (int i = 0; i < nttLen; i++)
    checksum += res[i];
  printf("Checksum: %lu\n", checksum);

  free(ntt);
  free(res);
  return 0;
}
