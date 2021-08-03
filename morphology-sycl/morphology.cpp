#include "morphology.h"


enum class MorphOpType {
    ERODE,
    DILATE,
};


template <MorphOpType opType>
inline unsigned char elementOp(unsigned char lhs, unsigned char rhs)
{
}

template <>
inline unsigned char elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
    return sycl::min(lhs, rhs);
}

template <>
inline unsigned char elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
    return sycl::max(lhs, rhs);
}


template <MorphOpType opType>
inline unsigned char borderValue()
{
}

template <>
inline unsigned char borderValue<MorphOpType::ERODE>()
{
    return BLACK;
}

template <>
inline unsigned char borderValue<MorphOpType::DILATE>()
{
    return WHITE;
}


template <MorphOpType opType>
void twoWayScan(unsigned char* __restrict__ sMem,
                unsigned char* __restrict__ opArray,
                const int selSize,
                const int tid,
                nd_item<2> &item)
{
    opArray[tid] = sMem[tid];
    opArray[tid + selSize] = sMem[tid + selSize];
    item.barrier(access::fence_space::local_space);

    for (int offset = 1; offset < selSize; offset *= 2) {
        if (tid >= offset) {
            opArray[tid + selSize - 1] = 
                elementOp<opType>(opArray[tid + selSize - 1], opArray[tid + selSize - 1 - offset]);
        }
        if (tid <= selSize - 1 - offset) {
            opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
        }
        item.barrier(access::fence_space::local_space);
    }
}


template <MorphOpType opType>
void vhgw_horiz(unsigned char* __restrict__ dst,
                const unsigned char* __restrict__ src,
                unsigned char* __restrict__ sMem,
                const int width,
                const int height,
                const int selSize,
                nd_item<2> &item)
{
    unsigned char* opArray = sMem + 2 * selSize;

    const int tidx = item.get_global_id(1);
    const int tidy = item.get_global_id(0);
    const int lidx = item.get_local_id(1);

    if (tidx >= width || tidy >= height) {
        return;
    }

    sMem[lidx] = src[tidy * width + tidx];
    if (tidx + selSize < width) {
        sMem[lidx + selSize] = src[tidy * width + tidx + selSize];
    }
    item.barrier(access::fence_space::local_space);

    twoWayScan<opType>(sMem, opArray, selSize, lidx, item);

    if (tidx + selSize/2 < width - selSize/2) {
        dst[tidy * width + tidx + selSize/2] = 
            elementOp<opType>(opArray[lidx], opArray[lidx + selSize - 1]);
    }
}

template <MorphOpType opType>
void vhgw_vert(unsigned char* __restrict__ dst,
               const unsigned char* __restrict__ src,
               unsigned char* __restrict__ sMem,
               const int width,
               const int height,
               const int selSize,
               nd_item<2> &item)
{
    unsigned char* opArray = sMem + 2 * selSize;

    const int tidx = item.get_global_id(1);
    const int tidy = item.get_global_id(0);
    const int lidy = item.get_local_id(0);

    if (tidy >= height || tidx >= width) {
        return;
    }

    sMem[lidy] = src[tidy * width + tidx];
    if (tidy + selSize < height) {
        sMem[lidy + selSize] = src[(tidy + selSize) * width + tidx];
    }
    item.barrier(access::fence_space::local_space);

    twoWayScan<opType>(sMem, opArray, selSize, lidy, item);

    if (tidy + selSize/2 < height - selSize/2) {
        dst[(tidy + selSize/2) * width + tidx] = 
            elementOp<opType>(opArray[lidy], opArray[lidy + selSize - 1]);
    }

    if (tidy < selSize/2 || tidy >= height - selSize/2) {
        dst[tidy * width + tidx] = borderValue<opType>();
    }
}


template <MorphOpType opType>
void morphology(
        queue &q,
        buffer<unsigned char> &img_d,
        buffer<unsigned char> &tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    int blockSize_x = hsize;
    int blockSize_y = 1;
    int gridSize_x = roundUp(width, blockSize_x);
    int gridSize_y = roundUp(height, blockSize_y);
    range<2> h_gws (gridSize_y * blockSize_y, gridSize_x * blockSize_x);
    range<2> h_lws (blockSize_y, blockSize_x);

    q.submit([&] (handler &cgh) {
      auto tmp = tmp_d.get_access<sycl_discard_write>(cgh);
      auto img = img_d.get_access<sycl_read>(cgh);
      accessor<unsigned char, 1, sycl_read_write, access::target::local> sMem(4*hsize, cgh);
      cgh.parallel_for<class horiz>(nd_range<2>(h_gws, h_lws), [=] (nd_item<2> item) {
        vhgw_horiz<opType>(tmp.get_pointer(), img.get_pointer(), sMem.get_pointer(),
                           width, height, hsize, item);
      });
    });

    blockSize_x = 1;
    blockSize_y = vsize;
    gridSize_x = roundUp(width, blockSize_x);
    gridSize_y = roundUp(height, blockSize_y);
    range<2> v_gws (gridSize_y * blockSize_y, gridSize_x * blockSize_x);
    range<2> v_lws (blockSize_y, blockSize_x);

    q.submit([&] (handler &cgh) {
      auto tmp = img_d.get_access<sycl_discard_write>(cgh);
      auto img = tmp_d.get_access<sycl_read>(cgh);
      accessor<unsigned char, 1, sycl_read_write, access::target::local> sMem(4*vsize, cgh);
      cgh.parallel_for<class vert>(nd_range<2>(v_gws, v_lws), [=] (nd_item<2> item) {
        vhgw_vert<opType>(tmp.get_pointer(), img.get_pointer(), sMem.get_pointer(), 
                          width, height, vsize, item);
      });
    });
}


extern "C"
void erode(
        queue &q,
        buffer<unsigned char,1> &img_d,
        buffer<unsigned char,1> &tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    morphology<MorphOpType::ERODE>(q, img_d, tmp_d, width, height, hsize, vsize);
}

extern "C"
void dilate(
        queue &q,
        buffer<unsigned char,1> &img_d,
        buffer<unsigned char,1> &tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    morphology<MorphOpType::DILATE>(q, img_d, tmp_d, width, height, hsize, vsize);
}
