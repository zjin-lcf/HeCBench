/*
This function reconstruct the 3D volume from projections, based on
the Distance-Driven principle. It works by calculating the overlap
in X and Y axis of the volume and the detector boundaries.
The geometry is for DBT with half cone-beam. All parameters are set
in "ParameterSettings" code.

Reference:
- Branchless Distance Driven Projection and Backprojection,
Samit Basu and Bruno De Man (2006)
- GPU Acceleration of Branchless Distance Driven Projection and
Backprojection, Liu et al (2016)
- GPU-Based Branchless Distance-Driven Projection and Backprojection,
Liu et al (2017)
- A GPU Implementation of Distance-Driven Computed Tomography,
Ryan D. Wagner (2017)
---------------------------------------------------------------------
Copyright (C) <2019>  <Rodrigo de Barros Vimieiro>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Original author: Rodrigo de Barros Vimieiro
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

using namespace sycl;

// thread block size
#define BLOCK_SIZE 256

// integration direction
#define integrateXcoord 1
#define integrateYcoord 0

void pad_projections_kernel(
    nd_item<3> &item,
    double* d_img,
    const int nDetXMap,
    const int nDetYMap,
    const int nElem,
    const int np)
{
  const int gid = item.get_global_id(2);
  if (gid < nElem)
    d_img[(np*nDetYMap *nDetXMap) + (gid*nDetYMap)] = 0;
}

void map_boudaries_kernel(
    nd_item<3> &item,
    double* d_pBound,
    const int nElem,
    const double valueLeftBound,
    const double sizeElem,
    const double offset)
{
  const int gid = item.get_global_id(2);
  if (gid < nElem)
    d_pBound[gid] = (gid - valueLeftBound) * sizeElem + offset;
}

void rot_detector_kernel(
    nd_item<3> &item,
          double* __restrict d_pRdetY,
          double* __restrict d_pRdetZ,
    const double* __restrict d_pYcoord,
    const double* __restrict d_pZcoord,
    const double yOffset,
    const double zOffset,
    const double phi,
    const int nElem)
{
  const int gid = item.get_global_id(2);
  if (gid < nElem) {
    // cos and sin are in measured in radians.
    d_pRdetY[gid] = ((d_pYcoord[gid] - yOffset) * sycl::cos(phi) - 
                     (d_pZcoord[gid] - zOffset) * sycl::sin(phi)) + yOffset;
    d_pRdetZ[gid] = ((d_pYcoord[gid] - yOffset) * sycl::sin(phi) +
                     (d_pZcoord[gid] - zOffset) * sycl::cos(phi)) + zOffset;
  }
}

void mapDet2Slice_kernel(
    nd_item<3> &item,
           double* __restrict const pXmapp,
           double* __restrict const pYmapp,
    double tubeX,
    double tubeY,
    double tubeZ,
    const double* __restrict const pXcoord,
    const double* __restrict const pYcoord,
    const double* __restrict const pZcoord,
    const double* __restrict const pZSlicecoord,
    const int nDetXMap,
    const int nDetYMap,
    const int nz)
{
  const int px = item.get_global_id(2);
  const int py = item.get_global_id(1);

  if (px < nDetYMap && py < nDetXMap) {

    const int pos = py * nDetYMap + px;

    pXmapp[pos] = ((pXcoord[py] - tubeX)*(pZSlicecoord[nz] - pZcoord[px]) - 
        (pXcoord[py] * tubeZ) + (pXcoord[py] * pZcoord[px])) / (-tubeZ + pZcoord[px]);

    if (py == 0)
      pYmapp[px] = ((pYcoord[px] - tubeY)*(pZSlicecoord[nz] - pZcoord[px]) -
          (pYcoord[px] * tubeZ) + (pYcoord[px] * pZcoord[px])) / (-tubeZ + pZcoord[px]);
  }
}

void img_integration_kernel(
    nd_item<3> &item,
    double* d_img,
    const int nPixX,
    const int nPixY,
    const bool direction,
    const int offsetX,
    const int offsetY,
    const int nSlices)
{
  /*
     Integration of 2D slices over the whole volume

     (S.1.Integration. - Liu et al(2017))

     Perform an inclusive scan
   */

  const int tx = item.get_global_id(2);
  const int ty = item.get_global_id(1);
  const int px = tx + offsetX;
  const int py = ty + offsetY;
  const int pz = item.get_global_id(0);

  if (px >= nPixY || py >= nPixX || pz >= nSlices) return;

  if (direction == integrateXcoord) {

    for (int s = 1; s <= item.get_local_range(1); s *= 2) {

      int spot = ty - s;

      double val = 0;

      if (spot >= 0) {
        val = d_img[(pz*nPixY*nPixX) + (offsetY + spot) * nPixY + px];
      }

      if (spot >= 0) {
        d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
      }
    }
  }
  else
  {
    for (int s = 1; s <= item.get_local_range(2); s *= 2) {

      int spot = tx - s;

      double val = 0;

      if (spot >= 0) {
        val = d_img[(pz*nPixY*nPixX) + py * nPixY + spot + offsetX];
      }

      if (spot >= 0) {
        d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
      }
    }
  }
}

void bilinear_interpolation_kernel(
    nd_item<3> &item,
          double* __restrict d_sliceI,
    const double* __restrict d_pProj,
    const double* __restrict d_pObjX,
    const double* __restrict d_pObjY,
    const double* __restrict d_pDetmX,
    const double* __restrict d_pDetmY,
    const int nPixXMap,
    const int nPixYMap,
    const int nDetXMap,
    const int nDetYMap,
    const int nDetX,
    const int nDetY,
    const int np) 
{
  const int px = item.get_global_id(2);
  const int py = item.get_global_id(1);

  // Make sure we don't try and access memory outside the detector
  // by having any threads mapped there return early
  if (px >= nPixYMap || py >= nPixXMap) return;

  //  S.2. Interpolation - Liu et al (2017)

  // Adjust the mapped coordinates to cross the range of (0-nDetX).*duMap 
  // Divide by pixelSize to get a unitary pixel size
  const double xNormData = nDetX - d_pObjX[py] / d_pDetmX[0];
  const int    xData = sycl::floor(xNormData);
  const double alpha = xNormData - xData;

  // Adjust the mapped coordinates to cross the range of (0-nDetY).*dyMap  
  // Divide by pixelSize to get a unitary pixel size
  const double yNormData = (d_pObjY[px] / d_pDetmX[0]) - (d_pDetmY[0] / d_pDetmX[0]);
  const int    yData = sycl::floor(yNormData);
  const double beta = yNormData - yData;

  double d00, d01, d10, d11;
  if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY)) 
    d00 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData)];
  else
    d00 = 0.0;

  if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY))
    d10 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData)];
  else
    d10 = 0.0;

  if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
    d01 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData + 1)];
  else
    d01 = 0.0;

  if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
    d11 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData + 1)];
  else
    d11 = 0.0;

  double result_temp1 = alpha * d10 + (-d00 * alpha + d00);
  double result_temp2 = alpha * d11 + (-d01 * alpha + d01);

  d_sliceI[py * nPixYMap + px] = beta * result_temp2 + (-result_temp1 * beta + result_temp1);
}

void differentiation_kernel(
    nd_item<3> &item,
          double* __restrict d_pVolume,
    const double* __restrict d_sliceI,
    double tubeX,
    double rtubeY,
    double rtubeZ,
    const double* __restrict const d_pObjX,
    const double* __restrict const d_pObjY,
    const double* __restrict const d_pObjZ,
    const int nPixX,
    const int nPixY,
    const int nPixXMap,
    const int nPixYMap,
    const double du,
    const double dv,
    const double dx,
    const double dy,
    const double dz,
    const int nz) 
{
  const int px = item.get_global_id(2);
  const int py = item.get_global_id(1);

  /*
     S.3. Differentiation - Eq. 24 - Liu et al (2017)

     Detector integral projection
     ___________
     |_A_|_B_|___|
     |_C_|_D_|___|
     |___|___|___|


     (px,py)
     ________________
     |_A_|__B__|_____|
     |_C_|(0,0)|(0,1)|
     |___|(1,0)|(1,1)|

     Threads are lauched from D up to nPixX (py) and nPixY (px)
     i.e., they are running on the detector image. Thread (0,0) is on D.

     Coordinates on intergal projection:

     A = py * nPixYMap + px
     B = ((py+1) * nPixYMap) + px
     C = py * nPixYMap + px + 1
     D = ((py+1) * nPixYMap) + px + 1
   */

  if (px < nPixY && py < nPixX) {

    const int pos = (nPixX*nPixY*nz) + (py * nPixY) + px;

    int coordA = py * nPixYMap + px;
    int coordB = ((py + 1) * nPixYMap) + px;
    int coordC = coordA + 1;
    int coordD = coordB + 1;

    // x - ray angle in X coord
    double gamma = sycl::atan((d_pObjX[py] + (dx / 2.0) - tubeX) / (rtubeZ - d_pObjZ[nz]));

    // x - ray angle in Y coord
    double alpha = sycl::atan((d_pObjY[px] + (dy / 2.0) - rtubeY) / (rtubeZ - d_pObjZ[nz]));

    double dA, dB, dC, dD;

    dA = d_sliceI[coordA];
    dB = d_sliceI[coordB];
    dC = d_sliceI[coordC];
    dD = d_sliceI[coordD];

    // Treat border of interpolated integral detector
    if (dC == 0 && dD == 0) {
      dC = dA;
      dD = dB;
    }

    // S.3.Differentiation - Eq. 24 - Liu et al(2017)
    d_pVolume[pos] += ((dD - dC - dB + dA)*(du*dv*dz / 
                       (sycl::cos(alpha)*sycl::cos(gamma)*dx*dy)));
  }
}

void division_kernel(
    nd_item<3> &item,
    double* d_img,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nProj)
{
  const int px = item.get_global_id(2);
  const int py = item.get_global_id(1);
  const int pz = item.get_global_id(0);
  if (px < nPixY && py < nPixX && pz < nSlices) {
    const int pos = (nPixX*nPixY*pz) + (py * nPixY) + px;
    d_img[pos] /= (double) nProj;
  }
}

// Branchless distance-driven backprojection 
void backprojectionDDb(
          double* const h_pVolume,
    const double* const h_pProj,
    const double* const h_pTubeAngle,
    const double* const h_pDetAngle,
    const int idXProj,
    const int nProj,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nDetX,
    const int nDetY,
    const double dx,
    const double dy,
    const double dz,
    const double du,
    const double dv,
    const double DSD,
    const double DDR,
    const double DAG)
{
  // Number of mapped detectors
  const int nDetXMap = nDetX + 1;
  const int nDetYMap = nDetY + 1;

  // Number of mapped pixels
  const int nPixXMap = nPixX + 1;
  const int nPixYMap = nPixY + 1;

#ifdef USE_GPU
  queue q(gpu_selector_v, property::queue::in_order());
#else
  queue q(cpu_selector_v, property::queue::in_order());
#endif

  double *d_pProj = malloc_device<double>(nDetXMap*nDetYMap*nProj, q);
  double *d_sliceI = malloc_device<double>(nPixXMap*nPixYMap, q);
  double *d_pVolume = malloc_device<double>(nPixX*nPixY*nSlices, q);

  // device memory for projections coordinates
  double *d_pDetX = malloc_device<double>( nDetXMap , q);
  double *d_pDetY = malloc_device<double>( nDetYMap , q);
  double *d_pDetZ = malloc_device<double>( nDetYMap , q);
  double *d_pObjX = malloc_device<double>( nPixXMap , q);
  double *d_pObjY = malloc_device<double>( nPixYMap , q);
  double *d_pObjZ = malloc_device<double>( nSlices , q);

  // device memory for mapped coordinates
  double *d_pDetmY = malloc_device<double>( nDetYMap , q);
  double *d_pDetmX = malloc_device<double>( nDetYMap * nDetXMap , q);

  // device memory for rotated detector coords
  double *d_pRdetY = malloc_device<double>( nDetYMap , q);
  double *d_pRdetZ = malloc_device<double>( nDetYMap , q);

  auto start = std::chrono::steady_clock::now();

  // Will reuse grid configurations
  range<3> gws (1,1,1);
  range<3> lws (1,1,1);

  const int maxThreadsPerBlock = BLOCK_SIZE;

  // Copy projection data padding with zeros for image integation

  // Initialize first column and row with zeros
  const double* h_pProj_tmp;
  double* d_pProj_tmp;

  lws[2] = maxThreadsPerBlock;
  gws[2] = (nDetXMap / maxThreadsPerBlock + 1) * maxThreadsPerBlock;

  for (int np = 0; np < nProj; np++) {

    // Pad on X coord direction
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class pad>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
        pad_projections_kernel (item, d_pProj,
                                nDetXMap, nDetYMap, nDetXMap, np);
      });
    });

    // Pad on Y coord direction
    d_pProj_tmp = d_pProj + (nDetXMap*nDetYMap*np) + 1;
    q.memset(d_pProj_tmp, 0, nPixY * sizeof(double));
  }

  // Copy projections data from host to device
  for (int np = 0; np < nProj; np++)
    for (int c = 0; c < nDetX; c++) {
      h_pProj_tmp = h_pProj + (c * nDetY) + (nDetX*nDetY*np);
      d_pProj_tmp = d_pProj + (((c + 1) * nDetYMap) + 1) + (nDetXMap*nDetYMap*np);
      q.memcpy(d_pProj_tmp, h_pProj_tmp, nDetY * sizeof(double));
    }

  // Generate detector and object boudaries

  lws[2] = maxThreadsPerBlock;

  gws[2] = (nDetX / maxThreadsPerBlock + 1) * maxThreadsPerBlock;

  q.submit([&] (handler &cgh) {
    cgh.parallel_for<class map_detX>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      map_boudaries_kernel(item, d_pDetX, nDetXMap, (double)nDetX, -du, 0.0);
    });
  });

  gws[2] = (nDetY / maxThreadsPerBlock + 1) * maxThreadsPerBlock;

  q.submit([&] (handler &cgh) {
    cgh.parallel_for<class map_detY>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      map_boudaries_kernel(item, d_pDetY, nDetYMap, nDetY / 2.0, dv, 0.0);
    });
  });

  gws[2] = (nPixX / maxThreadsPerBlock + 1) * maxThreadsPerBlock;

  q.submit([&] (handler &cgh) {
    cgh.parallel_for<class map_pixX>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      map_boudaries_kernel(item, d_pObjX, nPixXMap, (double)nPixX, -dx, 0.0);
    });
  });

  gws[2] = (nPixY / maxThreadsPerBlock + 1) * maxThreadsPerBlock;

  q.submit([&] (handler &cgh) {
    cgh.parallel_for<class map_pixY>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      map_boudaries_kernel(item, d_pObjY, nPixYMap, nPixY / 2.0, dy, 0.0);
    });
  });

  gws[2] = (nSlices / maxThreadsPerBlock + 1) * maxThreadsPerBlock;

  q.submit([&] (handler &cgh) {
    cgh.parallel_for<class map_pixZ>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      map_boudaries_kernel(item, d_pObjZ, nSlices, 0.0, dz, DAG + (dz / 2.0));
    });
  });

  // Initiate variables value with 0
  q.memset(d_pDetZ, 0, nDetYMap * sizeof(double));
  q.memset(d_pVolume, 0, nPixX * nPixY * nSlices * sizeof(double));

  // X - ray tube initial position
  double tubeX = 0;
  double tubeY = 0;
  double tubeZ = DSD;

  // Iso - center position
  double isoY = 0;
  double isoZ = DDR;

  // Integration of 2D projection over the whole projections
  // (S.1.Integration. - Liu et al(2017))

  // Naive integration o the X coord
  lws[2] = 8;
  lws[1] = 4;
  lws[0] = 8;

  gws[2] = 8 * (int)ceilf((float)nDetYMap / (8 - 1));
  gws[1] = 4;
  gws[0] = 8 * (int)ceilf((float)nProj / 8);

  int Xk = (int)ceilf((float)nDetXMap / (8 - 1));
  for (int k = 0; k < Xk; k++) {

    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class img_integralX>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
        img_integration_kernel(item, d_pProj,
          nDetXMap, nDetYMap, integrateXcoord, 0, k * 9, nProj);
      });
    });
  }

  // Naive integration o the Y coord
  lws[2] = 4;
  lws[1] = 8;
  lws[0] = 8;

  gws[2] = 4;
  gws[1] = 8 * (int)ceilf((float)nDetXMap / (8 - 1));
  gws[0] = 8 * (int)ceilf((float)nProj / 8);

  int Yk = (int)ceilf((float)nDetYMap / (8 - 1));
  for (int k = 0; k < Yk; k++) {

    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class img_integralY>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
        img_integration_kernel(item, d_pProj,
            nDetXMap, nDetYMap, integrateYcoord, k * 9, 0, nProj);
      });
    });
  }

  // loop over all projections ?
  int projIni, projEnd, nProj2Run;
  if (idXProj == -1) {
    projIni = 0;
    projEnd = nProj;
    nProj2Run = nProj;
  }
  else {
    projIni = idXProj;
    projEnd = idXProj + 1;
    nProj2Run = 1;
  }

  // For each projection
  for (int p = projIni; p < projEnd; p++) {

    // Get specif tube angle for the projection
    double theta = h_pTubeAngle[p] * M_PI / 180.0;

    // Get specif detector angle for the projection
    double phi = h_pDetAngle[p] * M_PI / 180.0;

    //printf("Tube angle:%f Det angle:%f\n", theta, phi);

    // Tube rotation
    double rtubeY = ((tubeY - isoY)*cos(theta) - (tubeZ - isoZ)*sin(theta)) + isoY;
    double rtubeZ = ((tubeY - isoY)*sin(theta) + (tubeZ - isoZ)*cos(theta)) + isoZ;

    //printf("R tube Y:%f R tube Z:%f\n", rtubeY, rtubeZ);

    // Detector rotation
    lws[2] = maxThreadsPerBlock;
    lws[1] = 1;
    lws[0] = 1;

    gws[2] = (nDetYMap / maxThreadsPerBlock + 1) * maxThreadsPerBlock;
    gws[1] = 1;
    gws[0] = 1;

    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class rot_det>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
        rot_detector_kernel(item, d_pRdetY, d_pRdetZ, d_pDetY, d_pDetZ,
                            isoY, isoZ, phi, nDetYMap);
      });
    });

    lws[2] = 16;
    lws[1] = 16;
    lws[0] = 1;

    // For each slice
    for (int nz = 0; nz < nSlices; nz++) {

      // Map detector onto XY plane(Inside proj loop in case detector rotates)

      gws[2] = (nDetYMap / 16 + 1) * 16;
      gws[1] = (nDetXMap / 16 + 1) * 16;
      gws[0] = 1;

      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class mapDet2Slice>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          mapDet2Slice_kernel (
            item,
            d_pDetmX, d_pDetmY, tubeX, rtubeY, rtubeZ, 
            d_pDetX, d_pRdetY, d_pRdetZ,
            d_pObjZ, nDetXMap, nDetYMap, nz);
        });
      });

      //  S.2. Interpolation - Liu et al (2017)

      gws[2] = (nPixYMap / 16 + 1) * 16;
      gws[1] = (nPixXMap / 16 + 1) * 16;

      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class bilinear_interp>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          bilinear_interpolation_kernel (item,
            d_sliceI, d_pProj,
            d_pObjX, d_pObjY, 
            d_pDetmX + nDetYMap * (nDetXMap-2), d_pDetmY,
            nPixXMap, nPixYMap, nDetXMap, nDetYMap, nDetX, nDetY, p);
        });
      });

      // S.3. Differentiation - Eq. 24 - Liu et al (2017)

      gws[2] = (nPixY / 16 + 1) * 16;
      gws[1] = (nPixX / 16 + 1) * 16;

      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class diff>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          differentiation_kernel (
            item, d_pVolume, d_sliceI, 
            tubeX, rtubeY, rtubeZ,
            d_pObjX, d_pObjY, d_pObjZ,
            nPixX, nPixY, nPixXMap, nPixYMap, du, dv, dx, dy, dz, nz);
        });
      });

    } // Loop end slices

  } // Loop end Projections


  // Normalize volume dividing by the number of projections
  lws[2] = 8;
  lws[1] = 8;
  lws[0] = 4;

  gws[2] = (nPixY / 8 + 1) * 8;
  gws[1] = (nPixX / 8 + 1) * 8;
  gws[0] = (nSlices / 4 + 1) * 4;

  q.submit([&] (handler &cgh) {
    cgh.parallel_for<class div>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      division_kernel (item, d_pVolume, 
                       nPixX, nPixY, nSlices, nProj2Run);
    });
  }).wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time %f (s)\n", time * 1e-9f);

  q.memcpy(h_pVolume, d_pVolume, nSlices* nPixX * nPixY * sizeof(double));

  free(d_pProj, q);
  free(d_sliceI, q);
  free(d_pVolume, q);
  free(d_pDetX, q);
  free(d_pDetY, q);
  free(d_pDetZ, q);
  free(d_pObjX, q);
  free(d_pObjY, q);
  free(d_pObjZ, q);
  free(d_pDetmY, q);
  free(d_pDetmX, q);
  free(d_pRdetY, q);
  free(d_pRdetZ, q);
}

int main() 
{
                            // image voxel density
  const int nPixX = 1996;   // number of voxels
  const int nPixY = 2457;   // number of voxels
  const int nSlices = 78;  

                            // detector panel pixel density
  const int nDetX = 1664;   // number of pixels
  const int nDetY = 2048;   // number of pixels

  const int nProj = 15;     // number of projections
  const int idXProj = -1;   // loop over all projections

  const double dx = 0.112;  // single voxel size (mm)
  const double dy = 0.112;
  const double dz = 1.0;

  const double du = 0.14;   // single detector size (mm)
  const double dv = 0.14;

  const double DSD = 700;   // distance from source to detector (mm)
  const double DDR = 0.0;   // distance from detector to pivot (mm)
  const double DAG = 25.0;  // distance of air gap (mm)

  const size_t pixVol = nPixX * nPixY * nSlices;
  const size_t detVol = nDetX * nDetY * nProj;
  double *h_pVolume = (double*) malloc (pixVol * sizeof(double));
  double *h_pProj = (double*) malloc (detVol * sizeof(double));

  double *h_pTubeAngle = (double*) malloc (nProj * sizeof(double));
  double *h_pDetAngle = (double*) malloc (nProj * sizeof(double));
  
  // tube angles in degrees
  for (int i = 0; i < nProj; i++) 
    h_pTubeAngle[i] = -7.5 + i * 15.0/nProj;

  // detector angles in degrees
  for (int i = 0; i < nProj; i++) 
    h_pDetAngle[i] = -2.1 + i * 4.2/nProj;

  // random values
  srand(123);
  for (size_t i = 0; i < pixVol; i++) 
    h_pVolume[i] = (double)rand() / (double)RAND_MAX;

  for (size_t i = 0; i < detVol; i++) 
    h_pProj[i] = (double)rand() / (double)RAND_MAX;

  backprojectionDDb(
    h_pVolume,
    h_pProj,
    h_pTubeAngle,
    h_pDetAngle,
    idXProj,
    nProj,
    nPixX, nPixY,
    nSlices,
    nDetX, nDetY,
    dx, dy, dz,
    du, dv,
    DSD, DDR, DAG);

  double checkSum = 0;
  for (size_t i = 0; i < pixVol; i++)
    checkSum += h_pVolume[i];
  printf("checksum = %lf\n", checkSum);

  free(h_pVolume);
  free(h_pTubeAngle);
  free(h_pDetAngle);
  free(h_pProj);
  return 0;
}
