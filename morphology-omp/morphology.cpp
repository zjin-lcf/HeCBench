#include "morphology.h"


enum class MorphOpType {
    ERODE,
    DILATE,
};


#pragma omp declare target
template <MorphOpType opType>
inline unsigned char elementOp(unsigned char lhs, unsigned char rhs)
{
}

template <>
inline unsigned char elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <>
inline unsigned char elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
    return lhs > rhs ? lhs : rhs;
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

// NOTE: step-efficient parallel scan
template <MorphOpType opType>
void twoWayScan(unsigned char* __restrict__ buffer,
        unsigned char* __restrict__ opArray,
        const int selSize,
        const int tid)
{
    opArray[tid] = buffer[tid];
    opArray[tid + selSize] = buffer[tid + selSize];
    #pragma omp barrier

    for (int offset = 1; offset < selSize; offset *= 2) {
        if (tid >= offset) {
            opArray[tid + selSize - 1] = 
               elementOp<opType>(opArray[tid + selSize - 1], opArray[tid + selSize - 1 - offset]);
        }
        if (tid <= selSize - 1 - offset) {
            opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
        }
        #pragma omp barrier
    }
}
#pragma omp end declare target


template <MorphOpType opType>
void morphology(
        const unsigned char* img_d,
        unsigned char* tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    int blockSize_x = hsize;
    int blockSize_y = 1;
    int gridSize_x = roundUp(width, blockSize_x);
    int gridSize_y = roundUp(height, blockSize_y);
    
    #pragma omp target teams num_teams(gridSize_x*gridSize_y) thread_limit(blockSize_x*blockSize_y)
    {
      //size_t sMemSize = 4 * hsize * sizeof(unsigned char);
      unsigned char sMem[128];
      #pragma omp parallel 
      {
        unsigned char* buffer = sMem;
        unsigned char* opArray = buffer + 2 * hsize;

        int bx = omp_get_team_num() % gridSize_x;
        int by = omp_get_team_num() / gridSize_x;
        int tx = omp_get_thread_num();   // blockSize_y = 1 so ty = 0

        const int tidx = tx + bx * blockSize_x;
        const int tidy =      by * blockSize_y;
        if (tidx < width && tidy < height) {

          buffer[tx] = img_d[tidy * width + tidx];
          if (tidx + hsize < width) {
              buffer[tx + hsize] = img_d[tidy * width + tidx + hsize];
          }
          #pragma omp barrier

          twoWayScan<opType>(buffer, opArray, hsize, tx);

          if (tidx + hsize/2 < width - hsize/2) {
              tmp_d[tidy * width + tidx + hsize/2] = 
                 elementOp<opType>(opArray[tx], opArray[tx + hsize - 1]);
          }
        }
      }
    }

    blockSize_x = 1;
    blockSize_y = vsize;
    gridSize_x = roundUp(width, blockSize_x);
    gridSize_y = roundUp(height, blockSize_y);

    #pragma omp target teams num_teams(gridSize_x*gridSize_y) thread_limit(blockSize_x*blockSize_y)
    {
      //size_t sMemSize = 4 * hsize * sizeof(unsigned char);
      unsigned char sMem[128];
      #pragma omp parallel 
      {
        unsigned char* buffer = sMem;
        unsigned char* opArray = buffer + 2 * vsize;

        int bx = omp_get_team_num() % gridSize_x;
        int by = omp_get_team_num() / gridSize_x;
        int ty = omp_get_thread_num();   // blockSize_x = 1 so tx = 0

        const int tidx =      bx * blockSize_x;
        const int tidy = ty + by * blockSize_y;
        if (tidx < width && tidy < height) {

          buffer[ty] = img_d[tidy * width + tidx];
          if (tidy + vsize < height) {
              buffer[ty + vsize] = img_d[(tidy + vsize) * width + tidx];
          }
          #pragma omp barrier

          twoWayScan<opType>(buffer, opArray, vsize, ty);

          if (tidy + vsize/2 < height - vsize/2) {
              tmp_d[(tidy + vsize/2) * width + tidx] = 
                  elementOp<opType>(opArray[ty], opArray[ty + vsize - 1]);
          }

          if (tidy < vsize/2 || tidy >= height - vsize/2) {
              tmp_d[tidy * width + tidx] = borderValue<opType>();
          }
        }
      }
    }
}


extern "C"
void erode(unsigned char* img_d,
        unsigned char* tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    morphology<MorphOpType::ERODE>(img_d, tmp_d, width, height, hsize, vsize);
}

extern "C"
void dilate(unsigned char* img_d,
        unsigned char* tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    morphology<MorphOpType::DILATE>(img_d, tmp_d, width, height, hsize, vsize);
}
