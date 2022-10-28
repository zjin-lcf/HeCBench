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
  nd_item<1> &item,
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
  nd_item<1> &item,
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
  queue &q,
  int numK,
  buffer<float, 1> &phiR_d,
  buffer<float, 1> &phiI_d,
  buffer<float, 1> &phiMag_d)
{
  int phiMagBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    phiMagBlocks++;

  range<1> lws (KERNEL_PHI_MAG_THREADS_PER_BLOCK);
  range<1> gws (KERNEL_PHI_MAG_THREADS_PER_BLOCK * phiMagBlocks);

  q.submit([&] (handler &cgh) {
    auto r = phiR_d.get_access<sycl_read>(cgh);
    auto i = phiI_d.get_access<sycl_read>(cgh);
    auto m = phiMag_d.get_access<sycl_write>(cgh);
    cgh.parallel_for<class compute_phi_mag>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      ComputePhiMag_GPU (
        item,
        numK,
        r.get_pointer(),
        i.get_pointer(),
        m.get_pointer());
    });
  });
}

void computeQ_GPU(
  queue &q,
  int numK, int numX,
  buffer<float, 1> &x_d,
  buffer<float, 1> &y_d,
  buffer<float, 1> &z_d,
  kValues          *kVals,
  buffer<float, 1> &Qr_d,
  buffer<float, 1> &Qi_d)
{
  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID) QGrids++;

  int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
  if (numX % KERNEL_Q_THREADS_PER_BLOCK) QBlocks++;

  range<1> lws (KERNEL_Q_THREADS_PER_BLOCK);
  range<1> gws (KERNEL_Q_THREADS_PER_BLOCK * QBlocks);

/* Values in the k-space coordinate system are stored in constant memory
 * on the GPU */
  buffer<kValues, 1> ck (KERNEL_Q_K_ELEMS_PER_GRID);

  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    q.submit([&] (handler &cgh) {
      auto acc = ck.get_access<sycl_write>(cgh, range<1>(numElems));
      cgh.copy(kValsTile, acc);
    });

    q.submit([&] (handler &cgh) {
      auto k = ck.get_access<sycl_read, sycl_cmem>(cgh);
      auto x = x_d.get_access<sycl_read>(cgh);
      auto y = y_d.get_access<sycl_read>(cgh);
      auto z = z_d.get_access<sycl_read>(cgh);
      auto Qr = Qr_d.get_access<sycl_read_write>(cgh);
      auto Qi = Qi_d.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class compute_q>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        ComputeQ_GPU (
          item,
          numK,
          QGridBase, 
          k.get_pointer(),
          x.get_pointer(),
          y.get_pointer(),
          z.get_pointer(),
          Qr.get_pointer(),
          Qi.get_pointer());
      });
    });
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}
