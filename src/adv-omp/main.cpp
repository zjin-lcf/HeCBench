#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>

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

dfloat *drandAlloc(int N){
  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));
  for(int n = 0; n < N; ++n) v[n] = drand48();
  return v;
}

// begin of advCubatureHex3D
void advCubatureHex3D (
    const int numTeams,
    const int numThreads,
    const int Nelements,
    const dfloat * __restrict vgeo,
    const dfloat * __restrict cubvgeo,
    const dfloat * __restrict cubD,
    const dfloat * __restrict cubInterpT,
    const int offset,
    const dfloat * __restrict U,
    dfloat * __restrict NU)
{
  #pragma omp target teams num_teams(numTeams)
  {
    dfloat s_cubD[16][16];
    dfloat s_cubInterpT[8][16];
    dfloat s_U[8][8];
    dfloat s_V[8][8];
    dfloat s_W[8][8];
    dfloat s_U1[16][16];
    dfloat s_V1[16][16];
    dfloat s_W1[16][16];
    #pragma omp parallel num_threads(numThreads)
    {
      dfloat r_U[16], r_V[16], r_W[16];
      dfloat r_Ud[16], r_Vd[16], r_Wd[16];

      const int e = omp_get_team_num();
      const int i = omp_get_thread_num() % 16;
      const int j = omp_get_thread_num() / 16;
      const int id = j * 16 + i;

      if (id < 8 * 16) s_cubInterpT[j][i] = cubInterpT[id];
      s_cubD[j][i] = cubD[id];

      for (int k = 0; k < 16; ++k) {
        r_U[k] = 0;
        r_V[k] = 0;
        r_W[k] = 0;
        r_Ud[k] = 0;
        r_Vd[k] = 0;
        r_Wd[k] = 0;
      }

      for (int c = 0; c < 8; ++c) {
        if (j < 8 && i < 8) {
          const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
          s_U[j][i] = U[id + 0 * offset];
          s_V[j][i] = U[id + 1 * offset];
          s_W[j][i] = U[id + 2 * offset];
        }

        #pragma omp barrier

        if (j < 8) {
          dfloat U1 = 0, V1 = 0, W1 = 0;
          for (int a = 0; a < 8; ++a) {
            dfloat Iia = s_cubInterpT[a][i];
            U1 += Iia * s_U[j][a];
            V1 += Iia * s_V[j][a];
            W1 += Iia * s_W[j][a];
          }
          s_U1[j][i] = U1;
          s_V1[j][i] = V1;
          s_W1[j][i] = W1;
        } else {
          s_U1[j][i] = 0;
          s_V1[j][i] = 0;
          s_W1[j][i] = 0;
        }

        #pragma omp barrier

        dfloat U2 = 0, V2 = 0, W2 = 0;
        for (int b = 0; b < 8; ++b) {
          dfloat Ijb = s_cubInterpT[b][j];
          U2 += Ijb * s_U1[b][i];
          V2 += Ijb * s_V1[b][i];
          W2 += Ijb * s_W1[b][i];
        }
        for (int k = 0; k < 16; ++k) {
          dfloat Ikc = s_cubInterpT[c][k];
          r_U[k] += Ikc * U2;
          r_V[k] += Ikc * V2;
          r_W[k] += Ikc * W2;
        }
        for (int k = 0; k < 16; ++k) {
          r_Ud[k] = r_U[k];
          r_Vd[k] = r_V[k];
          r_Wd[k] = r_W[k];
        }
      }

      for (int k = 0; k < 16; ++k) {
        s_U1[j][i] = r_Ud[k];
        s_V1[j][i] = r_Vd[k];
        s_W1[j][i] = r_Wd[k];

        #pragma omp barrier

        dfloat Udr = 0, Uds = 0, Udt = 0;
        dfloat Vdr = 0, Vds = 0, Vdt = 0;
        dfloat Wdr = 0, Wds = 0, Wdt = 0;
        for (int n = 0; n < 16; ++n) {
          dfloat Din = s_cubD[i][n];
          Udr += Din * s_U1[j][n];
          Vdr += Din * s_V1[j][n];
          Wdr += Din * s_W1[j][n];
        }
        for (int n = 0; n < 16; ++n) {
          dfloat Djn = s_cubD[j][n];
          Uds += Djn * s_U1[n][i];
          Vds += Djn * s_V1[n][i];
          Wds += Djn * s_W1[n][i];
        }
        for (int n = 0; n < 16; ++n) {
          dfloat Dkn = s_cubD[k][n];
          Udt += Dkn * r_Ud[n];
          Vdt += Dkn * r_Vd[n];
          Wdt += Dkn * r_Wd[n];
        }

        const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
        const dfloat drdx = cubvgeo[gid + p_RXID * p_cubNp];
        const dfloat drdy = cubvgeo[gid + p_RYID * p_cubNp];
        const dfloat drdz = cubvgeo[gid + p_RZID * p_cubNp];
        const dfloat dsdx = cubvgeo[gid + p_SXID * p_cubNp];
        const dfloat dsdy = cubvgeo[gid + p_SYID * p_cubNp];
        const dfloat dsdz = cubvgeo[gid + p_SZID * p_cubNp];
        const dfloat dtdx = cubvgeo[gid + p_TXID * p_cubNp];
        const dfloat dtdy = cubvgeo[gid + p_TYID * p_cubNp];
        const dfloat dtdz = cubvgeo[gid + p_TZID * p_cubNp];
        const dfloat JW = cubvgeo[gid + p_JWID * p_cubNp];
        const dfloat Un = r_U[k];
        const dfloat Vn = r_V[k];
        const dfloat Wn = r_W[k];
        const dfloat Uhat = JW * (Un * drdx + Vn * drdy + Wn * drdz);
        const dfloat Vhat = JW * (Un * dsdx + Vn * dsdy + Wn * dsdz);
        const dfloat What = JW * (Un * dtdx + Vn * dtdy + Wn * dtdz);
        r_U[k] = Uhat * Udr + Vhat * Uds + What * Udt;
        r_V[k] = Uhat * Vdr + Vhat * Vds + What * Vdt;
        r_W[k] = Uhat * Wdr + Vhat * Wds + What * Wdt;
      }

      for (int c = 0; c < 8; ++c) {
        dfloat rhsU = 0, rhsV = 0, rhsW = 0;
        for (int k = 0; k < 16; ++k) {
          dfloat Ikc = s_cubInterpT[c][k];
          rhsU += Ikc * r_U[k];
          rhsV += Ikc * r_V[k];
          rhsW += Ikc * r_W[k];
        }

        if (i < 8 && j < 8) {
          s_U[j][i] = rhsU;
          s_V[j][i] = rhsV;
          s_W[j][i] = rhsW;
        }

        #pragma omp barrier

        if (j < 8) {
          dfloat rhsU = 0, rhsV = 0, rhsW = 0;
          for (int k = 0; k < 16; ++k) {
            dfloat Ijb = s_cubInterpT[j][k];
            if (k < 8 && i < 8) {
              rhsU += Ijb * s_U[k][i];
              rhsV += Ijb * s_V[k][i];
              rhsW += Ijb * s_W[k][i];
            }
          }
          s_U1[j][i] = rhsU;
          s_V1[j][i] = rhsV;
          s_W1[j][i] = rhsW;
        }

        #pragma omp barrier

        if (i < 8 && j < 8) {
          dfloat rhsU = 0, rhsV = 0, rhsW = 0;
          for (int k = 0; k < 16; ++k) {
            dfloat Iia = s_cubInterpT[i][k];
            rhsU += Iia * s_U1[j][k];
            rhsV += Iia * s_V1[j][k];
            rhsW += Iia * s_W1[j][k];
          }
          const int gid = e * p_Np * p_Nvgeo + c * 8 * 8 + j * 8 + i;
          const dfloat IJW = vgeo[gid + p_IJWID * p_Np];
          const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
          NU[id + 0 * offset] = IJW * rhsU;
          NU[id + 1 * offset] = IJW * rhsV;
          NU[id + 2 * offset] = IJW * rhsW;
        }
      }
    }
  }
}
// end of advCubatureHex3D

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
  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);

  #pragma omp target data map(to: vgeo[0:Np*Nelements*p_Nvgeo], \
                                  cubvgeo[0:cubNp*Nelements*p_Nvgeo], \
                                  cubDiffInterpT[0:3*cubNp*Nelements], \
                                  cubInterpT[0:Np*cubNp], \
                                  u[0:3*Np*Nelements]) \
                          map(from: adv[0:3*Np*Nelements])

  {
    const int numTeams = Nelements;
    const int numThreads = 256;
    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    for(int test=0;test<Ntests;++test) {
      advCubatureHex3D(
        numTeams,
        numThreads,
        Nelements,
        vgeo,
        cubvgeo,
        cubDiffInterpT,
        cubInterpT,
        offset,
        u,
        adv);
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

    // statistics
    const dfloat GDOFPerSecond = (N*N*N)*Nelements/elapsed;
    std::cout << " NRepetitions=" << Ntests
              << " N=" << N
              << " cubN=" << cubN
              << " Nelements=" << Nelements
              << " elapsed time=" << elapsed
              << " GDOF/s=" << GDOFPerSecond
              << "\n";
  }

  double checksum = 0;
  for (int i = 0; i < 3*Np*Nelements; i++) {
    checksum += adv[i];
    #ifdef OUTPUT
    std::cout << adv[i] << "\n";
    #endif
  }
  std::cout << "Checksum=" << checksum << "\n";

  free(vgeo          );
  free(cubvgeo       );
  free(cubDiffInterpT);
  free(cubInterpT    );
  free(u             );
  free(adv           );
  return 0;
}
