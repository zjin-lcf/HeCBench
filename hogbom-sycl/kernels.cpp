#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include "kernels.h"
#include "common.h"

// grids and blocks are constant for the findPeak kernel
#define findPeakNBlocks 128
#define findPeakWidth 256

struct Peak {
  size_t pos;
  float val;
};

struct Position {
    Position(int _x, int _y) : x(_x), y(_y) { };
  int x;
  int y;
};

static Position idxToPos(const size_t idx, const int width)
{
  const int y = idx / width;
  const int x = idx % width;
  return Position(x, y);
}

static size_t posToIdx(const int width, const Position& pos)
{
  return (pos.y * width) + pos.x;
}

static Peak findPeak(queue &q, buffer<float> &d_image, buffer<Peak> &d_peak, size_t size)
{
  const int nBlocks = findPeakNBlocks;
  Peak peaks[nBlocks];

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that
  //buffer<Peak, 1> d_peak (nBlocks);

  range<1> gws (nBlocks * findPeakWidth);
  range<1> lws (findPeakWidth);

  // Find peak
  q.submit([&] (handler &cgh) {
    auto image = d_image.get_access<sycl_read>(cgh);
    auto absPeak = d_peak.get_access<sycl_read_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> maxVal(findPeakWidth, cgh);
    accessor<size_t, 1, sycl_read_write, access::target::local> maxPos(findPeakWidth, cgh);
    cgh.parallel_for<class find_peak>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      int tid = item.get_local_id(0);
      int bid = item.get_group(0);
      const int column = item.get_global_id(0);
      maxVal[tid] = 0.f;
      maxPos[tid] = 0;

      for (int idx = column; idx < size; idx += findPeakWidth*findPeakNBlocks) {
        if (sycl::fabs(image[idx]) > sycl::fabs(maxVal[tid])) {
          maxVal[tid] = image[idx];
          maxPos[tid] = idx;
        }
      }

      item.barrier(access::fence_space::local_space);

      if (tid == 0) {
        absPeak[bid].val = 0.f;
        absPeak[bid].pos = 0;
        for (int i = 0; i < findPeakWidth; ++i) {
          if (sycl::fabs(maxVal[i]) > sycl::fabs(absPeak[bid].val)) {
            absPeak[bid].val = maxVal[i];
            absPeak[bid].pos = maxPos[i];
          }
        }
      }
    });
  });

  // Get the peaks array back from the device
  q.submit([&] (handler &cgh) {
    auto acc = d_peak.get_access<sycl_read>(cgh);
    cgh.copy(acc, &peaks);
  }).wait();

  // Each thread block returned a peak, find the absolute maximum
  Peak p;
  p.val = 0.f;
  p.pos = 0;
  for (int i = 0; i < nBlocks; ++i) {
    if (fabsf(peaks[i].val) > fabsf(p.val)) {
      p.val = peaks[i].val;
      p.pos = peaks[i].pos;
    }
  }

  return p;
}

static void subtractPSF(
  queue &q,
  buffer<float> &d_psf, const int psfWidth,
  buffer<float> &d_residual, const int residualWidth,
  const size_t peakPos, const size_t psfPeakPos,
  const float absPeakVal, const float gain)
{
  const int blockDim = 16;

  const int rx = idxToPos(peakPos, residualWidth).x;
  const int ry = idxToPos(peakPos, residualWidth).y;

  const int px = idxToPos(psfPeakPos, psfWidth).x;
  const int py = idxToPos(psfPeakPos, psfWidth).y;

  const int diffx = rx - px;
  const int diffy = ry - px;

  const int startx = std::max(0, rx - px);
  const int starty = std::max(0, ry - py);

  const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
  const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

  // Note: Both start* and stop* locations are inclusive.
  const int blocksx = (stopx-startx + blockDim) / blockDim;
  const int blocksy = (stopy-starty + blockDim) / blockDim;

  range<2> gws (blocksy * blockDim, blocksx * blockDim);
  range<2> lws (blockDim, blockDim);
  q.submit([&] (handler &cgh) {
    auto psf = d_psf.get_access<sycl_read>(cgh);
    auto residual = d_residual.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class subtract_PSF>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      const int x = startx + item.get_global_id(1);
      const int y = starty + item.get_global_id(0);

      // thread blocks are of size 16, but the workload is not always a multiple of 16
      if (x <= stopx && y <= stopy) {
        residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
          * psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
      }
    });
  });
}

HogbomTest::HogbomTest()
{
}

HogbomTest::~HogbomTest()
{
}

void HogbomTest::deconvolve(const std::vector<float>& dirty,
    const size_t dirtyWidth,
    const std::vector<float>& psf,
    const size_t psfWidth,
    std::vector<float>& model,
    std::vector<float>& residual)
{
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  residual = dirty;

  // Initialise a peaks array on the device. Each thread block will return
  // a peak. Note:  the d_peaks array is not initialized (hence avoiding the
  // memcpy), it is up to the device function to do that
  buffer<Peak, 1> d_peaks (findPeakNBlocks);

  const size_t psf_size = psf.size();
  const size_t residual_size = residual.size();
  buffer<float, 1> d_psf (&psf[0], psf_size);
  buffer<float, 1> d_residual (&residual[0], residual_size);

  // Find peak of PSF
  Peak psfPeak = findPeak(q, d_psf, d_peaks, psf_size);

  std::cout << "Found peak of PSF: " << "Maximum = " << psfPeak.val 
    << " at location " << idxToPos(psfPeak.pos, psfWidth).x << ","
    << idxToPos(psfPeak.pos, psfWidth).y << std::endl;
  assert(psfPeak.pos <= psf_size);

  for (unsigned int i = 0; i < niters; ++i) {
    // Find peak in the residual image
    Peak peak = findPeak(q, d_residual, d_peaks, residual_size);

    assert(peak.pos <= residual_size);

    // Check if threshold has been reached
    if (fabsf(peak.val) < threshold) {
      std::cout << "Reached stopping threshold" << std::endl;
    }

    // Subtract the PSF from the residual image 
    subtractPSF(q, d_psf, psfWidth, d_residual, dirtyWidth, peak.pos, psfPeak.pos, peak.val, gain);

    // Add to model
    model[peak.pos] += peak.val * gain;
  }
}
