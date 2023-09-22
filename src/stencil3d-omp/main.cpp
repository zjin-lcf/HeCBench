#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

// 2D block size
#define BSIZE 16
// Tile size in the x direction
#define XTILE 20

typedef float Real;

void stencil3d(
    const Real*__restrict d_psi, 
          Real*__restrict d_npsi, 
    const Real*__restrict d_sigmaX, 
    const Real*__restrict d_sigmaY, 
    const Real*__restrict d_sigmaZ,
    int bdimx, int bdimy, int bdimz,
    int nx, int ny, int nz)
{
  #pragma omp target teams num_teams(bdimz*bdimy*bdimx) thread_limit(BSIZE*BSIZE)
  {
    Real sm_psi[4][BSIZE][BSIZE];
    #pragma omp parallel 
    {
      #define V0(y,z) sm_psi[pii][y][z]
      #define V1(y,z) sm_psi[cii][y][z]
      #define V2(y,z) sm_psi[nii][y][z]
      
      #define sigmaX(x,y,z,dir) d_sigmaX[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
      #define sigmaY(x,y,z,dir) d_sigmaY[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
      #define sigmaZ(x,y,z,dir) d_sigmaZ[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
      
      #define psi(x,y,z) d_psi[ z + nz * ( (y) + ny * (x) ) ]
      #define npsi(x,y,z) d_npsi[ z + nz * ( (y) + ny * (x) ) ]

      const int tjj = omp_get_thread_num() / BSIZE;
      const int tkk = omp_get_thread_num() % BSIZE;
      const int blockIdx_x = omp_get_team_num() % bdimx;
      const int blockIdx_y = omp_get_team_num() / bdimx % bdimy;
      const int blockIdx_z = omp_get_team_num()  / (bdimx * bdimy);
      const int gridDim_x = bdimx;
      const int gridDim_y = bdimy;
      const int gridDim_z = bdimz;

      // shift for each tile by updating device pointers
      d_psi = &(psi(XTILE*blockIdx_x, (BSIZE-2)*blockIdx_y, (BSIZE-2)*blockIdx_z));
      d_npsi = &(npsi(XTILE*blockIdx_x, (BSIZE-2)*blockIdx_y, (BSIZE-2)*blockIdx_z));

      d_sigmaX = &(sigmaX(XTILE*blockIdx_x, (BSIZE-2)*blockIdx_y, (BSIZE-2)*blockIdx_z, 0));
      d_sigmaY = &(sigmaY(XTILE*blockIdx_x, (BSIZE-2)*blockIdx_y, (BSIZE-2)*blockIdx_z, 0));
      d_sigmaZ = &(sigmaZ(XTILE*blockIdx_x, (BSIZE-2)*blockIdx_y, (BSIZE-2)*blockIdx_z, 0));

      int nLast_x=XTILE+1; int nLast_y=(BSIZE-1); int nLast_z=(BSIZE-1);
      if (blockIdx_x == gridDim_x-1) nLast_x = nx-2 - XTILE * blockIdx_x + 1;
      if (blockIdx_y == gridDim_y-1) nLast_y = ny-2 - (BSIZE-2) * blockIdx_y + 1;
      if (blockIdx_z == gridDim_z-1) nLast_z = nz-2 - (BSIZE-2) * blockIdx_z + 1;

      // previous, current, next, and temp indices
      int pii,cii,nii,tii;
      Real xcharge,ycharge,zcharge,dV = 0;

      if(tjj <= nLast_y && tkk <= nLast_z) {
        pii=0; cii=1; nii=2;
        sm_psi[cii][tjj][tkk] = psi(0,tjj,tkk);
        sm_psi[nii][tjj][tkk] = psi(1,tjj,tkk);
      }

      #pragma omp barrier

      //initial
      if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y))
      {
        Real xd=-V1(tjj,tkk) + V2(tjj,tkk);
        Real yd=(-V1(-1 + tjj,tkk) + V1(1 + tjj,tkk) - V2(-1 + tjj,tkk) + V2(1 + tjj,tkk))/4.;
        Real zd=(-V1(tjj,-1 + tkk) + V1(tjj,1 + tkk) - V2(tjj,-1 + tkk) + V2(tjj,1 + tkk))/4.;
        dV -= sigmaX(1,tjj,tkk,0) * xd + sigmaX(1,tjj,tkk,1) * yd + sigmaX(1,tjj,tkk,2) * zd ; 
      }

      if(tjj <= nLast_y && tkk <= nLast_z) {
        tii=pii; pii=cii; cii=nii; nii=tii;
      }

      for(int ii=1;ii<nLast_x;ii++)
      {
        if(tjj <= nLast_y && tkk <= nLast_z)
          sm_psi[nii][tjj][tkk] = psi(ii+1,tjj,tkk);
        #pragma omp barrier

        // y face current
        if ((tkk>0) && (tkk<nLast_z) && (tjj<nLast_y))
        {
          Real xd=(-V0(tjj,tkk) - V0(1 + tjj,tkk) + V2(tjj,tkk) + V2(1 + tjj,tkk))/4.;
          Real yd=-V1(tjj,tkk) + V1(1 + tjj,tkk);
          Real zd=(-V1(tjj,-1 + tkk) + V1(tjj,1 + tkk) - V1(1 + tjj,-1 + tkk) + V1(1 + tjj,1 + tkk))/4.;
          ycharge = sigmaY(ii,tjj+1,tkk,0) * xd + sigmaY(ii,tjj+1,tkk,1) * yd + sigmaY(ii,tjj+1,tkk,2) * zd ; 
          dV += ycharge;
          sm_psi[3][tjj][tkk]=ycharge;
        }
        #pragma omp barrier

        if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y))
          dV -= sm_psi[3][tjj-1][tkk];  //bring from left

        #pragma omp barrier

        // z face current
        if ((tkk<nLast_z) && (tjj>0) && (tjj<nLast_y))
        {
          Real xd=(-V0(tjj,tkk) - V0(tjj,1 + tkk) + V2(tjj,tkk) + V2(tjj,1 + tkk))/4.;
          Real yd=(-V1(-1 + tjj,tkk) - V1(-1 + tjj,1 + tkk) + V1(1 + tjj,tkk) + V1(1 + tjj,1 + tkk))/4.;
          Real zd=-V1(tjj,tkk) + V1(tjj,1 + tkk);
          zcharge = sigmaZ(ii,tjj,tkk+1,0) * xd + sigmaZ(ii,tjj,tkk+1,1) * yd + sigmaZ(ii,tjj,tkk+1,2) * zd ; 
          dV += zcharge;
          sm_psi[3][tjj][tkk]=zcharge;
        }

        #pragma omp barrier

        if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y))
          dV -= sm_psi[3][tjj][tkk-1];
        #pragma omp barrier

        // x face current
        if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y))
        {
          Real xd=-V1(tjj,tkk) + V2(tjj,tkk);
          Real yd=(-V1(-1 + tjj,tkk) + V1(1 + tjj,tkk) - V2(-1 + tjj,tkk) + V2(1 + tjj,tkk))/4.;
          Real zd=(-V1(tjj,-1 + tkk) + V1(tjj,1 + tkk) - V2(tjj,-1 + tkk) + V2(tjj,1 + tkk))/4.;
          xcharge = sigmaX(ii+1,tjj,tkk,0) * xd + sigmaX(ii+1,tjj,tkk,1) * yd + sigmaX(ii+1,tjj,tkk,2) * zd ; 
          dV += xcharge;
          npsi(ii,tjj,tkk) = dV; //store dV
          dV = -xcharge; //pass to the next cell in x-dir
        }
        #pragma omp barrier
        if(tjj <= nLast_y && tkk <= nLast_z) {
          tii=pii; pii=cii; cii=nii; nii=tii;
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <grid dimension> <repeat>\n", argv[0]);
    return 1;
  }
  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int nx = size;
  const int ny = size;
  const int nz = size;
  const int vol = nx * ny * nz;
  printf("Grid dimension: nx=%d ny=%d nz=%d\n",nx,ny,nz);

  // allocate and initialize Vm
  Real *h_Vm = (Real*)malloc(sizeof(Real)*vol);

  #define h_Vm(x,y,z) h_Vm[ z + nz * ( y + ny * ( x  ) ) ]

  for(int ii=0;ii<nx;ii++)
    for(int jj=0;jj<ny;jj++)
      for(int kk=0;kk<nz;kk++)
        h_Vm(ii,jj,kk) = (ii*(ny*nz) + jj * nz + kk) % 19;

  // allocate and initialize sigma
  Real *h_sigma = (Real*) malloc(sizeof(Real)*vol*9);
  for (int i = 0; i < vol*9; i++) h_sigma[i] = i % 19;

  // reset dVm
  Real *h_dVm = (Real*) malloc (sizeof(Real) * vol);
  memset(h_dVm, 0, sizeof(Real) * vol);

  //determine block sizes
  int bdimz = (nz-2)/(BSIZE-2) + ((nz-2)%(BSIZE-2)==0?0:1);
  int bdimy = (ny-2)/(BSIZE-2) + ((ny-2)%(BSIZE-2)==0?0:1);
  int bdimx = (nx-2)/XTILE + ((nx-2)%XTILE==0?0:1);

  #pragma omp target data map(to: h_Vm[0:vol], h_sigma[0:vol*9]) map(tofrom: h_dVm[0:vol])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      stencil3d(h_Vm, h_dVm, h_sigma, h_sigma + 3*vol, h_sigma + 6*vol, 
                bdimx, bdimy, bdimz, nx, ny, nz);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

#ifdef DUMP
  for(int ii=0;ii<nx;ii++)
    for(int jj=0;jj<ny;jj++)
      for(int kk=0;kk<nz;kk++)
        printf("dVm (%d,%d,%d)=%e\n",ii,jj,kk,h_dVm[kk+nz*(jj+ny*ii)]);
#endif

  free(h_sigma);
  free(h_Vm);
  free(h_dVm);

  return 0;
}
