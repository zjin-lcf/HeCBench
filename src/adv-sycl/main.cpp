#include <iostream>
#include <cstdlib>
#include <chrono>
#include <sycl/sycl.hpp>

#define p_IJWID 6
#define p_JID   4
#define p_JWID  5
#define p_Np    512
#define p_Nq    8
#define p_Nvgeo 12
#define p_RXID  0
#define p_RYID  1
#define p_RZID  7
#define p_SXID  2
#define p_SYID  3
#define p_SZID  8
#define p_TXID  9
#define p_TYID  10
#define p_TZID  11
#define p_cubNp 4096
#define p_cubNq 16

// kernel
#include "adv.h"

dfloat *drandAlloc(int N){
  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));
  for(int n = 0; n < N; ++n) v[n] = drand48();
  return v;
}

dfloat *deviceAlloc(sycl::queue &q, const dfloat *h, int N){
  dfloat *d = sycl::malloc_device<dfloat>(N, q);
  q.memcpy(d, h, N * sizeof(dfloat));
  return d;
}

int main(int argc, char **argv) {

  if (argc < 4) {
    printf("Usage: ./adv N cubN numElements [nRepetitions]\n");
    exit(-1);
  }

  const int N = atoi(argv[1]);
  const int cubN = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);
  int Ntests = 1;

  if(argc >= 5) Ntests = atoi(argv[4]);

  const int Nq = N+1;
  const int cubNq = cubN+1;
  const int Np = Nq*Nq*Nq;
  const int cubNp = cubNq*cubNq*cubNq;
  const dlong offset = Nelements*Np;

  printf("Data type in bytes: %zu\n", sizeof(dfloat));

  srand48(123);
  dfloat *h_vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *h_cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *h_cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *h_cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *h_u              = drandAlloc(3*Np*Nelements);
  dfloat *h_adv            = drandAlloc(3*Np*Nelements);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  dfloat *vgeo           = deviceAlloc(q, h_vgeo, Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = deviceAlloc(q, h_cubvgeo, cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = deviceAlloc(q, h_cubDiffInterpT, 3*cubNp*Nelements);
  dfloat *cubInterpT     = deviceAlloc(q, h_cubInterpT, Np*cubNp);
  dfloat *u              = deviceAlloc(q, h_u, 3*Np*Nelements);
  dfloat *adv            = deviceAlloc(q, h_adv, 3*Np*Nelements);

  sycl::range<3> gws (1, 16, Nelements*16);
  sycl::range<3> lws (1, 16, 16);

  q.wait();
  auto start = std::chrono::high_resolution_clock::now();

  // run kernel
  for(int test=0;test<Ntests;++test) {
    advCubatureHex3D(q,
                     gws,
                     lws,
                     0,
                     Nelements,
                     vgeo,
                     cubvgeo,
                     cubDiffInterpT,
                     cubInterpT,
                     offset,
                     u,
                     adv);
  }

  q.wait();
  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

  q.memcpy(h_adv, adv, 3*Np*Nelements*sizeof(dfloat)).wait();

  sycl::free(vgeo          , q);
  sycl::free(cubvgeo       , q);
  sycl::free(cubDiffInterpT, q);
  sycl::free(cubInterpT    , q);
  sycl::free(u             , q);
  sycl::free(adv           , q);

  double checksum = 0;
  for (int i = 0; i < 3*Np*Nelements; i++) {
    checksum += h_adv[i];
    #ifdef OUTPUT
    std::cout << h_adv[i] << "\n";
    #endif
  }
  std::cout << "Checksum=" << checksum << "\n";

  // statistics
  const dfloat GDOFPerSecond = (N*N*N)*Nelements/elapsed;
  std::cout << " NRepetitions=" << Ntests
            << " N=" << N
            << " cubN=" << cubN
            << " Nelements=" << Nelements
            << " elapsed time=" << elapsed
            << " GDOF/s=" << GDOFPerSecond
            << "\n";

  free(h_vgeo          );
  free(h_cubvgeo       );
  free(h_cubDiffInterpT);
  free(h_cubInterpT    );
  free(h_u             );
  free(h_adv           );
  return 0;
}
