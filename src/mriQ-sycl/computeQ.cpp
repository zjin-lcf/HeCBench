/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 256
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

void ComputePhiMag_GPU(
  sycl::nd_item<1> &item,
  const int numK,
  const float* __restrict phiR,
  const float* __restrict phiI,
        float* __restrict phiMag)
{
  int indexK = item.get_global_id(0);
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

void ComputeQ_GPU(
  sycl::nd_item<1> &item,
  const int numK,
        int kGlobalIndex,
  const kValues* __restrict ck,
  const float* __restrict x,
  const float* __restrict y,
  const float* __restrict z,
        float* __restrict Qr,
        float* __restrict Qi)
{
  // Determine the element of the X arrays computed by this thread
  int xIndex = item.get_global_id(0);

  // Read block's X values from global mem to shared mem
  float sX = x[xIndex];
  float sY = y[xIndex];
  float sZ = z[xIndex];
  float sQr = Qr[xIndex];
  float sQi = Qi[xIndex];

  // Loop over all elements of K in constant mem to compute a partial value
  // for X.
  int kIndex = 0;
  if (numK % 2) {
    float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
    sQr += ck[0].PhiMag * sycl::cos(expArg);
    sQi += ck[0].PhiMag * sycl::sin(expArg);
    kIndex++;
    kGlobalIndex++;
  }

  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex += 2, kGlobalIndex += 2) {
    float expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
    sQr += ck[kIndex].PhiMag * sycl::cos(expArg);
    sQi += ck[kIndex].PhiMag * sycl::sin(expArg);

    int kIndex1 = kIndex + 1;
    float expArg1 = PIx2 * (ck[kIndex1].Kx * sX +
			    ck[kIndex1].Ky * sY +
			    ck[kIndex1].Kz * sZ);
    sQr += ck[kIndex1].PhiMag * sycl::cos(expArg1);
    sQi += ck[kIndex1].PhiMag * sycl::sin(expArg1);
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
}

void computePhiMag_GPU(
  sycl::queue &q,
  int numK,
  float *phiR_d,
  float *phiI_d,
  float *phiMag_d)
{
  int phiMagBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    phiMagBlocks++;

  sycl::range<1> lws (KERNEL_PHI_MAG_THREADS_PER_BLOCK);
  sycl::range<1> gws (KERNEL_PHI_MAG_THREADS_PER_BLOCK * phiMagBlocks);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class compute_phi_mag>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      ComputePhiMag_GPU (
        item,
        numK,
        phiR_d,
        phiI_d,
        phiMag_d);
    });
  });
}

void computeQ_GPU(
  sycl::queue &q,
  int numK, int numX,
  float *x_d,
  float *y_d,
  float *z_d,
  kValues *kVals,
  float *Qr_d,
  float *Qi_d)
{
  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID) QGrids++;

  int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
  if (numX % KERNEL_Q_THREADS_PER_BLOCK) QBlocks++;

  sycl::range<1> lws (KERNEL_Q_THREADS_PER_BLOCK);
  sycl::range<1> gws (KERNEL_Q_THREADS_PER_BLOCK * QBlocks);

/* Values in the k-space coordinate system are stored in constant memory
 * on the GPU */
  kValues *ck = sycl::malloc_device<kValues>(KERNEL_Q_K_ELEMS_PER_GRID, q);

  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    q.memcpy(ck, kValsTile, numElems * sizeof(kValues));

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class compute_q>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        ComputeQ_GPU (
          item,
          numK,
          QGridBase,
          ck,
          x_d,
          y_d,
          z_d,
          Qr_d,
          Qi_d);
      });
    });
  }
  q.wait();
  sycl::free(ck, q);
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}
