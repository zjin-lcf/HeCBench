#include <math.h>

void reference(
    const   int *__restrict d_tisspoints,
    const float *__restrict d_gtt,
    const float *__restrict d_gbartt,
          float *__restrict d_ct,
    const float *__restrict d_ctprev,
    const float *__restrict d_qt,
    int nnt, int nntDev, int step, int isp)
{
  for (int i = 0; i < step * nnt; i++) {
    int jtp,ixyz,ix,iy,iz,jx,jy,jz,istep;
    int nnt2 = 2*nnt;
    float p = 0.f;

    int itp = i/step;
    int itp1 = i%step;
    if(itp < nnt) {
      ix = d_tisspoints[itp];
      iy = d_tisspoints[itp+nnt];
      iz = d_tisspoints[itp+nnt2];
      for(jtp=itp1; jtp<nnt; jtp+=step){
        jx = d_tisspoints[jtp];
        jy = d_tisspoints[jtp+nnt];
        jz = d_tisspoints[jtp+nnt2];
        ixyz = abs(jx-ix) + abs(jy-iy) + abs(jz-iz) + (isp-1)*nntDev;
        p += d_gtt[ixyz]*d_ctprev[jtp] + d_gbartt[ixyz]*d_qt[jtp];
      }
      if(itp1 == 0) d_ct[itp] = p;
    }

    for(istep=1; istep<step; istep++)
      if(itp1 == istep && itp < nnt) d_ct[itp] += p;
  }
}


