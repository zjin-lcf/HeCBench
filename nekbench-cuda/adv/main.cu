#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <cuda.h>

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

// gpu kernel
#include "advKernel-opt.cpp"

dfloat *drandAlloc(int N){
  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));
  for(int n=0;n<N;++n){
    v[n] = drand48();
  }
  return v;
}

int main(int argc, char **argv){

  if(argc<4){
    printf("Usage: ./adv N cubN numElements [nRepetitions]\n");
    return 1;
  }

  const int N = atoi(argv[1]);
  const int cubN = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);

  int Ntests = 1;
  if(argc>=5)
    Ntests = atoi(argv[4]);

  const int Nq = N+1;
  const int cubNq = cubN+1;
  const int Np = Nq*Nq*Nq;
  const int cubNp = cubNq*cubNq*cubNq;

  const dlong offset = Nelements*Np;

  std::cout << "word size: " << sizeof(dfloat) << " bytes\n";

  srand48(2);

  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);

  auto start = std::chrono::high_resolution_clock::now();

  dfloat *o_vgeo, *o_cubvgeo, *o_cubDiffInterpT, *o_cubInterpT, *o_u, *o_adv;
  cudaMalloc((void**)&o_vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat));
  cudaMalloc((void**)&o_cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat));
  cudaMalloc((void**)&o_cubDiffInterpT,3*cubNp*Nelements*sizeof(dfloat));
  cudaMalloc((void**)&o_cubInterpT, Np*cubNp*sizeof(dfloat));
  cudaMalloc((void**)&o_u, 3*Np*Nelements*sizeof(dfloat));
  cudaMalloc((void**)&o_adv, 3*Np*Nelements*sizeof(dfloat));

  cudaMemcpy(o_vgeo, vgeo, Np*Nelements*p_Nvgeo*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_cubvgeo, cubvgeo, cubNp*Nelements*p_Nvgeo*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_cubDiffInterpT, cubDiffInterpT, 3*cubNp*Nelements*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_cubInterpT, cubInterpT, Np*cubNp*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_u, u, 3*Np*Nelements*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_adv, adv, 3*Np*Nelements*sizeof(dfloat), cudaMemcpyHostToDevice);

  // run kernel
  for(int test=0;test<Ntests;++test) 
    advCubatureHex3D <<< dim3(Nelements, 1), dim3(16, 16) >>> (
        Nelements,
        o_vgeo,
        o_cubvgeo,
        o_cubDiffInterpT,
        o_cubInterpT,
        offset,
        o_u,
        o_adv);

  cudaMemcpy(adv, o_adv, 3*Np*Nelements*sizeof(dfloat), cudaMemcpyDeviceToHost);

  cudaFree(o_vgeo);
  cudaFree(o_cubvgeo);
  cudaFree(o_cubDiffInterpT);
  cudaFree(o_cubInterpT);
  cudaFree(o_u);
  cudaFree(o_adv);

  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

#ifdef DEBUG
  for (int i = 0; i < 3*Np*Nelements; i++) std::cout << adv[i] << "\n";
#endif

  // print statistics
  const dfloat GDOFPerSecond = (N*N*N)*Nelements/elapsed;
  std::cout << " NRepetitions=" << Ntests
            << " N=" << N
            << " cubN=" << cubN
            << " Nelements=" << Nelements
            << " elapsed time=" << elapsed
            << " GDOF/s=" << GDOFPerSecond
            << "\n";
  return 0;
}

