//
// An implementation of Parallel Marching Blocks algorithm
//

#include <cstdio>
#include <random>
#include "common.h"
#include "tables.h"

// problem size
constexpr unsigned int N(1024);
constexpr unsigned int Nd2(N / 2);
constexpr unsigned int voxelXLv1(16);
constexpr unsigned int voxelYLv1(16);
constexpr unsigned int voxelZLv1(64);
constexpr unsigned int gridXLv1((N - 1) / (voxelXLv1 - 1));
constexpr unsigned int gridYLv1((N - 1) / (voxelYLv1 - 1));
constexpr unsigned int gridZLv1((N - 1) / (voxelZLv1 - 1));
constexpr unsigned int countingThreadNumLv1(128);
constexpr unsigned int blockNum(gridXLv1* gridYLv1* gridZLv1);
constexpr unsigned int countingBlockNumLv1(blockNum / countingThreadNumLv1);

constexpr unsigned int voxelXLv2(4);
constexpr unsigned int voxelYLv2(4);
constexpr unsigned int voxelZLv2(8);
constexpr unsigned int blockXLv2(5);
constexpr unsigned int blockYLv2(5);
constexpr unsigned int blockZLv2(9);
constexpr unsigned int voxelNumLv2(blockXLv2* blockYLv2* blockZLv2);

constexpr unsigned int countingThreadNumLv2(1024);
constexpr unsigned int gridXLv2(gridXLv1* blockXLv2);
constexpr unsigned int gridYLv2(gridYLv1* blockYLv2);
//constexpr unsigned int gridZLv2(gridZLv1* blockZLv2);

inline float f(unsigned int x, unsigned int y, unsigned int z)
{
  constexpr float d(2.0f / N);
  float xf((int(x - Nd2)) * d);//[-1, 1)
  float yf((int(z - Nd2)) * d);
  float zf((int(z - Nd2)) * d);
  return 1.f - 16.f * xf * yf * zf - 4.f * (xf * xf + yf * yf + zf * zf);
}

inline float zeroPoint(unsigned int x, float v0, float v1, float isoValue)
{
  return ((x * (v1 - isoValue) + (x + 1) * (isoValue - v0)) / (v1 - v0) - Nd2) * (2.0f / N);
}

inline float transformToCoord(unsigned int x)
{
  return (int(x) - int(Nd2)) * (2.0f / N);
}

void computeMinMaxLv1(float*__restrict minMax, float *__restrict sminMax, nd_item<3> &item)
{
  constexpr unsigned int threadNum(voxelXLv1 * voxelYLv1);
  constexpr unsigned int warpNum(threadNum / 32);
  auto sg = item.get_sub_group();
  int blockIdx_z = item.get_group(0);
  int blockIdx_y = item.get_group(1);
  int blockIdx_x = item.get_group(2);
  int threadIdx_z = item.get_local_id(0);
  int threadIdx_y = item.get_local_id(1);
  int threadIdx_x = item.get_local_id(2);
  unsigned int x(blockIdx_x * (voxelXLv1 - 1) + threadIdx_x);
  unsigned int y(blockIdx_y * (voxelYLv1 - 1) + threadIdx_y);
  unsigned int z(blockIdx_z * (voxelZLv1 - 1));
  unsigned int tid(threadIdx_x + voxelXLv1 * threadIdx_y);
  unsigned int laneid = tid % 32;
  unsigned int blockid(blockIdx_x + gridXLv1 * (blockIdx_y + gridYLv1 * blockIdx_z));
  unsigned int warpid(tid >> 5);
  float v(f(x, y, z));
  float minV(v), maxV(v);
  for (int c0(1); c0 < voxelZLv1; ++c0)
  {
    v = f(x, y, z + c0);
    if (v < minV)minV = v;
    if (v > maxV)maxV = v;
  }
#pragma unroll
  for (int c0(16); c0 > 0; c0 /= 2)
  {
    float t0, t1;
    t0 = sg.shuffle_down(minV, c0);
    t1 = sg.shuffle_down(maxV, c0);
    if (t0 < minV)minV = t0;
    if (t1 > maxV)maxV = t1;
  }
  if (laneid == 0)
  {
    sminMax[warpid] = minV;
    sminMax[warpid + warpNum] = maxV;
  }
  item.barrier(access::fence_space::local_space);
  if (warpid == 0)
  {
    minV = sminMax[laneid];
    maxV = sminMax[laneid + warpNum];
#pragma unroll
    for (int c0(warpNum / 2); c0 > 0; c0 /= 2)
    {
      float t0, t1;
      t0 = sg.shuffle_down(minV, c0);
      t1 = sg.shuffle_down(maxV, c0);
      if (t0 < minV)minV = t0;
      if (t1 > maxV)maxV = t1;
    }
    if (laneid == 0)
    {
      minMax[blockid * 2] = minV;
      minMax[blockid * 2 + 1] = maxV;
    }
  }
}

void compactLv1(
  float isoValue, 
  const float*__restrict minMax,
  unsigned int*__restrict blockIndices,
  unsigned int*__restrict countedBlockNum,
  unsigned int*__restrict sums,
  nd_item<1> &item)
{
  auto sg = item.get_sub_group();
  constexpr unsigned int warpNum(countingThreadNumLv1 / 32);
  unsigned int tid(item.get_local_id(0));
  unsigned int laneid = tid % 32;
  unsigned int bIdx(item.get_group(0) * countingThreadNumLv1 + tid);
  unsigned int warpid(tid >> 5);
  unsigned int test;
  if (minMax[2 * bIdx] <= isoValue && minMax[2 * bIdx + 1] >= isoValue)test = 1;
  else test = 0;
  unsigned int testSum(test);
#pragma unroll
  for (int c0(1); c0 < 32; c0 *= 2)
  {
    unsigned int tp(sg.shuffle_up(testSum, c0));
    if (laneid >= c0) testSum += tp;
  }
  if (laneid == 31)sums[warpid] = testSum;
  item.barrier(access::fence_space::local_space);
  if (warpid == 0)
  {
    unsigned warpSum = sums[laneid];
#pragma unroll
    for (int c0(1); c0 < warpNum; c0 *= 2)
    {
      unsigned int tp(sg.shuffle_up(warpSum, c0));
      if (laneid >= c0) warpSum += tp;
    }
    sums[laneid] = warpSum;
  }
  item.barrier(access::fence_space::local_space);
  if (warpid != 0)testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv1 - 1 && testSum != 0) {
    //sums[31] = atomicAdd(countedBlockNum, testSum);
    auto atomic_obj_ref = ONEAPI::atomic_ref<unsigned int, 
                          ONEAPI::memory_order::relaxed,
                          ONEAPI::memory_scope::device,
                          access::address_space::global_space> (countedBlockNum[0]);
    sums[31] = atomic_obj_ref.fetch_add(testSum);
  }
  item.barrier(access::fence_space::local_space);
  if (test) blockIndices[testSum + sums[31] - 1] = bIdx;
}

void computeMinMaxLv2(
  const unsigned int*__restrict blockIndicesLv1,
  float*__restrict minMax,
  nd_item<2> &item)
{
  auto sg = item.get_sub_group();
  unsigned int tid(item.get_local_id(1));
  unsigned int voxelOffset(item.get_local_id(0));
  unsigned int blockIdx_x = item.get_group(1);
  unsigned int blockIndex(blockIndicesLv1[blockIdx_x]);
  unsigned int tp(blockIndex);
  unsigned int x((blockIndex % gridXLv1) * (voxelXLv1 - 1) + (voxelOffset % 5) * (voxelXLv2 - 1) + (tid & 3));
  tp /= gridXLv1;
  unsigned int y((tp % gridYLv1) * (voxelYLv1 - 1) + (voxelOffset / 5) * (voxelYLv2 - 1) + (tid >> 2));
  tp /= gridYLv1;
  unsigned int z(tp * (voxelZLv1 - 1));
  float v(f(x, y, z));
  float minV(v), maxV(v);
  unsigned int idx(2 * (voxelOffset + voxelNumLv2 * blockIdx_x));
  for (int c0(0); c0 < blockZLv2; ++c0)
  {
    for (int c1(1); c1 < voxelZLv2; ++c1)
    {
      v = f(x, y, z + c1);
      if (v < minV)minV = v;
      if (v > maxV)maxV = v;
    }
    z += voxelZLv2 - 1;
#pragma unroll
    for (int c1(8); c1 > 0; c1 /= 2)
    {
      float t0, t1;
      t0 = sg.shuffle_down(minV, c1);
      t1 = sg.shuffle_down(maxV, c1);
      if (t0 < minV)minV = t0;
      if (t1 > maxV)maxV = t1;
    }
    if (tid == 0)
    {
      minMax[idx] = minV;
      minMax[idx + 1] = maxV;
      constexpr unsigned int offsetSize(2 * blockXLv2 * blockYLv2);
      idx += offsetSize;
    }
    minV = v;
    maxV = v;
  }
}

void compactLv2(
  float isoValue,
  const float*__restrict minMax,
  const unsigned int*__restrict blockIndicesLv1,
  unsigned int*__restrict blockIndicesLv2,
  unsigned int counterBlockNumLv1,
  unsigned int*__restrict countedBlockNumLv2,
  unsigned int*__restrict sums,
  nd_item<1> &item)
{
  auto sg = item.get_sub_group();
  constexpr unsigned int warpNum(countingThreadNumLv2 / 32);
  unsigned int tid(item.get_local_id(0));
  unsigned int laneid = tid % 32;
  unsigned int warpid(tid >> 5);
  unsigned int id0(tid + item.get_group(0) * countingThreadNumLv2);
  unsigned int id1(id0 / voxelNumLv2);
  unsigned int test;
  if (id1 < counterBlockNumLv1)
  {
    if (minMax[2 * id0] <= isoValue && minMax[2 * id0 + 1] >= isoValue)
      test = 1;
    else
      test = 0;
  }
  else test = 0;
  unsigned int testSum(test);
#pragma unroll
  for (int c0(1); c0 < 32; c0 *= 2)
  {
    unsigned int tp(sg.shuffle_up(testSum, c0));
    if (laneid >= c0) testSum += tp;
  }
  if (laneid == 31) sums[warpid] = testSum;
  item.barrier(access::fence_space::local_space);
  if (warpid == 0)
  {
    unsigned int warpSum = sums[laneid];
#pragma unroll
    for (int c0(1); c0 < warpNum; c0 *= 2)
    {
      unsigned int tp(sg.shuffle_up(warpSum, c0));
      if (laneid >= c0)warpSum += tp;
    }
    sums[laneid] = warpSum;
  }
  item.barrier(access::fence_space::local_space);
  if (warpid != 0) testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv2 - 1) {
    //sums[31] = atomicAdd(countedBlockNumLv2, testSum);
    auto atomic_obj_ref = ONEAPI::atomic_ref<unsigned int, 
                          ONEAPI::memory_order::relaxed,
                          ONEAPI::memory_scope::device,
                          access::address_space::global_space> (countedBlockNumLv2[0]);
    sums[31] = atomic_obj_ref.fetch_add(testSum);
  }
  item.barrier(access::fence_space::local_space);

  if (test)
  {
    unsigned int bIdx1(blockIndicesLv1[id1]);
    unsigned int bIdx2;
    unsigned int x1, y1, z1;
    unsigned int x2, y2, z2;
    unsigned int tp1(bIdx1);
    unsigned int tp2((tid + item.get_group(0) * countingThreadNumLv2) % voxelNumLv2);
    x1 = tp1 % gridXLv1;
    x2 = tp2 % blockXLv2;
    tp1 /= gridXLv1;
    tp2 /= blockXLv2;
    y1 = tp1 % gridYLv1;
    y2 = tp2 % blockYLv2;
    z1 = tp1 / gridYLv1;
    z2 = tp2 / blockYLv2;
    bIdx2 = x2 + blockXLv2 * (x1 + gridXLv1 * (y2 + blockYLv2 * (y1 + gridYLv1 * (z1 * blockZLv2 + z2))));
    blockIndicesLv2[testSum + sums[31] - 1] = bIdx2;
  }
}

int main(int argc, char* argv[])
{
  unsigned int iterations = atoi(argv[1]);

  std::uniform_real_distribution<float>rd(0, 1);
  std::mt19937 mt(123);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> minMaxLv1Device ( blockNum * 2 );
  buffer<unsigned int, 1> blockIndicesLv1Device ( blockNum );
  buffer<unsigned int, 1> countedBlockNumLv1Device (1);
  buffer<unsigned int, 1> countedBlockNumLv2Device (1);
  buffer<unsigned short, 1> distinctEdgesTableDevice (distinctEdgesTable, 256);
  buffer<int, 1> triTableDevice (triTable, 256*16);
  buffer<uchar4, 1> edgeIDTableDevice (edgeIDTable, 12);
  buffer<unsigned int, 1> countedVerticesNumDevice (1);
  buffer<unsigned int, 1> countedTrianglesNumDevice (1);

  // simulate rendering without memory allocation for vertices and triangles 
  buffer<unsigned long long, 1> trianglesDevice (1);
  buffer<float, 1> coordXDevice (1);
  buffer<float, 1> coordYDevice (1);
  buffer<float, 1> coordZDevice (1);
  buffer<float, 1> coordZPDevice (1);

  const range<3> BlockSizeLv1{ 1, voxelYLv1, voxelXLv1};
  const range<3> GridSizeLv1{ gridZLv1, gridYLv1, gridXLv1 };
  const range<2> BlockSizeLv2{ blockXLv2 * blockYLv2, voxelXLv2 * voxelYLv2 };
  const range<3> BlockSizeGenerating{ voxelZLv2, voxelYLv2, voxelXLv2 };

  float isoValue(-0.9f);

  unsigned int countedBlockNumLv1;
  unsigned int countedBlockNumLv2;
  unsigned int countedVerticesNum;
  unsigned int countedTrianglesNum;

  for (unsigned int c0(0); c0 < iterations; ++c0)
  {
    q.wait();

    q.submit([&] (handler &cgh) {
      auto acc = countedBlockNumLv1Device.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0u);
    });

    q.submit([&] (handler &cgh) {
      auto acc = countedBlockNumLv2Device.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0u);
    });

    q.submit([&] (handler &cgh) {
      auto acc = countedVerticesNumDevice.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0u);
    });

    q.submit([&] (handler &cgh) {
      auto acc = countedTrianglesNumDevice.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0u);
    });

    q.submit([&] (handler &cgh) {
      auto acc = trianglesDevice.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0ull);
    });

    q.submit([&] (handler &cgh) {
      auto acc = coordXDevice.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0.f);
    });

    q.submit([&] (handler &cgh) {
      auto acc = coordYDevice.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0.f);
    });

    q.submit([&] (handler &cgh) {
      auto acc = coordZDevice.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0.f);
    });

    q.submit([&] (handler &cgh) {
      auto acc = coordZPDevice.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0.f);
    });

    q.submit([&] (handler &cgh) {
      auto minMaxLv1 = minMaxLv1Device.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> smem(64, cgh);
      cgh.parallel_for<class min_max1>(nd_range<3>(GridSizeLv1*BlockSizeLv1, BlockSizeLv1), [=] (nd_item<3> item) {
        computeMinMaxLv1(minMaxLv1.get_pointer(), smem.get_pointer(), item);
      });
    });

    q.submit([&] (handler &cgh) {
      auto minMaxLv1 = minMaxLv1Device.get_access<sycl_read>(cgh);
      auto blockIndicesLv1 = blockIndicesLv1Device.get_access<sycl_discard_write>(cgh);
      auto countedBlockNumLv1 = countedBlockNumLv1Device.get_access<sycl_discard_write>(cgh);
      accessor<unsigned int, 1, sycl_read_write, access::target::local> smem(32, cgh);
      cgh.parallel_for<class compact1>(nd_range<1>(
        range<1>(countingBlockNumLv1*countingThreadNumLv1), 
        range<1>(countingThreadNumLv1)), [=] (nd_item<1> item) {
        compactLv1(isoValue, 
                     minMaxLv1.get_pointer(),
                     blockIndicesLv1.get_pointer(),
                     countedBlockNumLv1.get_pointer(),
                     smem.get_pointer(),
                     item);
      });
    });

    q.submit([&] (handler &cgh) {
      auto acc = countedBlockNumLv1Device.get_access<sycl_read>(cgh);
      cgh.copy(acc, &countedBlockNumLv1);
    }).wait();

    buffer<float, 1> minMaxLv2Device (countedBlockNumLv1 * voxelNumLv2 * 2);

    const range<2> GridSizeLv2 (1, countedBlockNumLv1);

    q.submit([&] (handler &cgh) {
      auto blockIndicesLv1 = blockIndicesLv1Device.get_access<sycl_read>(cgh);
      auto minMaxLv2 = minMaxLv2Device.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class min_max2>(nd_range<2>(GridSizeLv2*BlockSizeLv2, BlockSizeLv2), [=] (nd_item<2> item) {
        computeMinMaxLv2(blockIndicesLv1.get_pointer(), minMaxLv2.get_pointer(), item);
      });
    });

    buffer<unsigned int, 1> blockIndicesLv2Device (countedBlockNumLv1 * voxelNumLv2);
    unsigned int countingBlockNumLv2((countedBlockNumLv1 * voxelNumLv2 + countingThreadNumLv2 - 1) / countingThreadNumLv2);

    q.submit([&] (handler &cgh) {
      auto minMaxLv2 = minMaxLv2Device.get_access<sycl_read>(cgh);
      auto blockIndicesLv1 = blockIndicesLv1Device.get_access<sycl_read>(cgh);
      auto blockIndicesLv2 = blockIndicesLv2Device.get_access<sycl_discard_write>(cgh);
      auto countedBlockNumLv2 = countedBlockNumLv2Device.get_access<sycl_discard_write>(cgh);
      accessor<unsigned int, 1, sycl_read_write, access::target::local> smem(32, cgh);
      cgh.parallel_for<class compact2>(nd_range<1>(
        range<1>(countingBlockNumLv2*countingThreadNumLv2),
        range<1>(countingThreadNumLv2)), [=] (nd_item<1> item) {
        compactLv2(isoValue, 
                     minMaxLv2.get_pointer(),
                     blockIndicesLv1.get_pointer(),
                     blockIndicesLv2.get_pointer(),
                     countedBlockNumLv1,
                     countedBlockNumLv2.get_pointer(),
                     smem.get_pointer(),
                     item);
      });
    });

    q.submit([&] (handler &cgh) {
      auto acc = countedBlockNumLv2Device.get_access<sycl_read>(cgh);
      cgh.copy(acc, &countedBlockNumLv2);
    }).wait();

    q.submit([&] (handler &cgh) {
      auto blockIndicesLv2 = blockIndicesLv2Device.get_access<sycl_read>(cgh);
      auto distinctEdgesTable = distinctEdgesTableDevice.get_access<sycl_read>(cgh); 
      auto triTable = triTableDevice.get_access<sycl_read>(cgh); 
      auto edgeIDTable = edgeIDTableDevice.get_access<sycl_read>(cgh);
      auto countedVerticesNum = countedVerticesNumDevice.get_access<sycl_read_write>(cgh); 
      auto countedTrianglesNum = countedTrianglesNumDevice.get_access<sycl_read_write>(cgh); 
      auto triangles = trianglesDevice.get_access<sycl_read_write>(cgh);
      auto coordX = coordXDevice.get_access<sycl_read_write>(cgh); 
      auto coordY = coordYDevice.get_access<sycl_read_write>(cgh); 
      auto coordZ = coordZDevice.get_access<sycl_read_write>(cgh); 
      auto coordZP = coordZPDevice.get_access<sycl_read_write>(cgh);

      accessor<unsigned short, 3, sycl_read_write, access::target::local> vertexIndices({voxelZLv2, voxelYLv2, voxelXLv2}, cgh);
      accessor<float, 3, sycl_read_write, access::target::local> value({voxelZLv2+1, voxelYLv2+1, voxelXLv2+1}, cgh);
      accessor<unsigned int, 1, sycl_read_write, access::target::local> sumsVertices(32, cgh);
      accessor<unsigned int, 1, sycl_read_write, access::target::local> sumsTriangles(32, cgh);

      const range<3> trianglesBlock (1,1,countedBlockNumLv2);
      cgh.parallel_for<class triangles_gen>(
        nd_range<3>(trianglesBlock*BlockSizeGenerating, BlockSizeGenerating), [=] (nd_item<3> item) {
        auto sg = item.get_sub_group();
        unsigned int threadIdx_x = item.get_local_id(2);
        unsigned int threadIdx_y = item.get_local_id(1);
        unsigned int threadIdx_z = item.get_local_id(0);

        unsigned int blockId(blockIndicesLv2[item.get_group(2)]);
        unsigned int tp(blockId);
        unsigned int x((tp % gridXLv2) * (voxelXLv2 - 1) + threadIdx_x);
        tp /= gridXLv2;
        unsigned int y((tp % gridYLv2) * (voxelYLv2 - 1) + threadIdx_y);
        unsigned int z((tp / gridYLv2) * (voxelZLv2 - 1) + threadIdx_z);
        unsigned int eds(7);
        float v(value[threadIdx_z][threadIdx_y][threadIdx_x] = f(x, y, z));
        if (threadIdx_x == voxelXLv2 - 1)
        {
          eds &= 6;
          value[threadIdx_z][threadIdx_y][voxelXLv2] = f(x + 1, y, z);
          if (threadIdx_y == voxelYLv2 - 1)
            value[threadIdx_z][voxelYLv2][voxelXLv2] = f(x + 1, y + 1, z);
        }
        if (threadIdx_y == voxelYLv2 - 1)
        {
          eds &= 5;
          value[threadIdx_z][voxelYLv2][threadIdx_x] = f(x, y + 1, z);
          if (threadIdx_z == voxelZLv2 - 1)
            value[voxelZLv2][voxelYLv2][threadIdx_x] = f(x, y + 1, z + 1);
        }
        if (threadIdx_z == voxelZLv2 - 1)
        {
          eds &= 3;
          value[voxelZLv2][threadIdx_y][threadIdx_x] = f(x, y, z + 1);
          if (threadIdx_x == voxelXLv2 - 1)
            value[voxelZLv2][threadIdx_y][voxelXLv2] = f(x + 1, y, z + 1);
        }
        eds <<= 13;
        item.barrier(access::fence_space::local_space);
        unsigned int cubeCase(0);
        if (value[threadIdx_z][threadIdx_y][threadIdx_x] < isoValue) cubeCase |= 1;
        if (value[threadIdx_z][threadIdx_y][threadIdx_x + 1] < isoValue) cubeCase |= 2;
        if (value[threadIdx_z][threadIdx_y + 1][threadIdx_x + 1] < isoValue) cubeCase |= 4;
        if (value[threadIdx_z][threadIdx_y + 1][threadIdx_x] < isoValue) cubeCase |= 8;
        if (value[threadIdx_z + 1][threadIdx_y][threadIdx_x] < isoValue) cubeCase |= 16;
        if (value[threadIdx_z + 1][threadIdx_y][threadIdx_x + 1] < isoValue) cubeCase |= 32;
        if (value[threadIdx_z + 1][threadIdx_y + 1][threadIdx_x + 1] < isoValue) cubeCase |= 64;
        if (value[threadIdx_z + 1][threadIdx_y + 1][threadIdx_x] < isoValue) cubeCase |= 128;

        unsigned int distinctEdges(eds ? distinctEdgesTable[cubeCase] : 0);
        unsigned int numTriangles(eds != 0xe000 ? 0 : distinctEdges & 7);
        unsigned int numVertices(sycl::popcount(distinctEdges &= eds));
        unsigned int laneid = (threadIdx_x + voxelXLv2 * (threadIdx_y + voxelYLv2 * threadIdx_z)) % 32;
        unsigned warpid((threadIdx_x + voxelXLv2 * (threadIdx_y + voxelYLv2 * threadIdx_z)) >> 5);
        constexpr unsigned int threadNum(voxelXLv2 * voxelYLv2 * voxelZLv2);
        constexpr unsigned int warpNum(threadNum / 32);
        unsigned int sumVertices(numVertices);
        unsigned int sumTriangles(numTriangles);

        #pragma unroll
        for (int c0(1); c0 < 32; c0 *= 2)
        {
          unsigned int tp0(sg.shuffle_up(sumVertices, c0));
          unsigned int tp1(sg.shuffle_up(sumTriangles, c0));
          if (laneid >= c0)
          {
            sumVertices += tp0;
            sumTriangles += tp1;
          }
        }
        if (laneid == 31)
        {
          sumsVertices[warpid] = sumVertices;
          sumsTriangles[warpid] = sumTriangles;
        }
        item.barrier(access::fence_space::local_space);
        if (warpid == 0)
        {
          unsigned warpSumVertices = sumsVertices[laneid];
          unsigned warpSumTriangles = sumsTriangles[laneid];
          #pragma unroll
          for (int c0(1); c0 < warpNum; c0 *= 2)
          {
            unsigned int tp0(sg.shuffle_up(warpSumVertices, c0));
            unsigned int tp1(sg.shuffle_up(warpSumTriangles, c0));
            if (laneid >= c0)
            {
              warpSumVertices += tp0;
              warpSumTriangles += tp1;
            }
          }
          sumsVertices[laneid] = warpSumVertices;
          sumsTriangles[laneid] = warpSumTriangles;
        }
        item.barrier(access::fence_space::local_space);
        if (warpid != 0)
        {
          sumVertices += sumsVertices[warpid - 1];
          sumTriangles += sumsTriangles[warpid - 1];
        }
        if (eds == 0)
        {
          //sumsVertices[31] = atomicAdd(countedVerticesNum, sumVertices);
          //sumsTriangles[31] = atomicAdd(countedTrianglesNum, sumTriangles);
          auto cv_ref = ONEAPI::atomic_ref<unsigned int, 
            ONEAPI::memory_order::relaxed,
            ONEAPI::memory_scope::device,
            access::address_space::global_space> (countedVerticesNum[0]);

          auto ct_ref = ONEAPI::atomic_ref<unsigned int, 
            ONEAPI::memory_order::relaxed,
            ONEAPI::memory_scope::device,
            access::address_space::global_space> (countedTrianglesNum[0]);

          sumsVertices[31] = cv_ref.fetch_add(sumVertices);
          sumsTriangles[31] = ct_ref.fetch_add(sumTriangles);
        }

        unsigned int interOffsetVertices(sumVertices - numVertices);
        sumVertices = interOffsetVertices + sumsVertices[31];//exclusive offset
        sumTriangles = sumTriangles + sumsTriangles[31] - numTriangles;//exclusive offset
        vertexIndices[threadIdx_z][threadIdx_y][threadIdx_x] = interOffsetVertices | distinctEdges;
        item.barrier(access::fence_space::local_space);

        for (unsigned int c0(0); c0 < numTriangles; ++c0)
        {
          #pragma unroll
          for (unsigned int c1(0); c1 < 3; ++c1)
          {
            int edgeID(triTable[16 * cubeCase + 3 * c0 + c1]);
            uchar4 edgePos(edgeIDTable[edgeID]);
            unsigned short vertexIndex(
              vertexIndices[threadIdx_z + edgePos.z()][threadIdx_y + edgePos.y()][threadIdx_x + edgePos.x()]);
            unsigned int tp(sycl::popcount(vertexIndex >> (16 - edgePos.w())) + (vertexIndex & 0x1fff));
            //atomicAdd(triangles, (unsigned long long)(sumsVertices[31] + tp));
            auto triangles_ref = ONEAPI::atomic_ref<unsigned long long, 
                  ONEAPI::memory_order::relaxed,
                  ONEAPI::memory_scope::device,
                  access::address_space::global_space> (triangles[0]);
            triangles_ref.fetch_add((unsigned long long)(sumsVertices[31] + tp));
          }
        }

        // sumVertices may be too large for a GPU memory
        float zp = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;

        if (distinctEdges & (1 << 15))
        {
          zp = zeroPoint(x, v, value[threadIdx_z][threadIdx_y][threadIdx_x + 1], isoValue);
          cy = transformToCoord(y);
          cz = transformToCoord(z);
        }
        if (distinctEdges & (1 << 14))
        {
          cx = transformToCoord(x);
          zp += zeroPoint(y, v, value[threadIdx_z][threadIdx_y + 1][threadIdx_x], isoValue);
          cz += transformToCoord(z);
        }
        if (distinctEdges & (1 << 13))
        {
          cx += transformToCoord(x);
          cy += transformToCoord(y);
          zp += zeroPoint(z, v, value[threadIdx_z + 1][threadIdx_y][threadIdx_x], isoValue);
        }
        //atomicAdd(coordX, cx);
        //atomicAdd(coordY, cy);
        //atomicAdd(coordZ, cz);
        //atomicAdd(coordZP, zp);
        auto x_ref = ONEAPI::atomic_ref<float, 
          ONEAPI::memory_order::relaxed,
          ONEAPI::memory_scope::device,
          access::address_space::global_space> (coordX[0]);
        x_ref.fetch_add(cx);

        auto y_ref = ONEAPI::atomic_ref<float, 
          ONEAPI::memory_order::relaxed,
          ONEAPI::memory_scope::device,
          access::address_space::global_space> (coordY[0]);
        y_ref.fetch_add(cy);

        auto z_ref = ONEAPI::atomic_ref<float, 
          ONEAPI::memory_order::relaxed,
          ONEAPI::memory_scope::device,
          access::address_space::global_space> (coordZ[0]);
        z_ref.fetch_add(cz);

        auto zp_ref = ONEAPI::atomic_ref<float, 
          ONEAPI::memory_order::relaxed,
          ONEAPI::memory_scope::device,
          access::address_space::global_space> (coordZP[0]);
        zp_ref.fetch_add(zp);
      });
    });

    q.submit([&] (handler &cgh) {
      auto acc = countedVerticesNumDevice.get_access<sycl_read>(cgh);
      cgh.copy(acc, &countedVerticesNum);
    }).wait();

    q.submit([&] (handler &cgh) {
      auto acc = countedTrianglesNumDevice.get_access<sycl_read>(cgh);
      cgh.copy(acc, &countedTrianglesNum);
    }).wait();

  }

  printf("Block Lv1: %u\nBlock Lv2: %u\n", countedBlockNumLv1, countedBlockNumLv2);
  printf("Vertices Size: %u\n", countedBlockNumLv2 * 304);
  printf("Triangles Size: %u\n", countedBlockNumLv2 * 315 * 3);
  printf("Vertices: %u\nTriangles: %u\n", countedVerticesNum, countedTrianglesNum);

  // specific to the problem size
  bool ok = (countedBlockNumLv1 == 8296 && countedBlockNumLv2 == 240380 &&
             countedVerticesNum == 4856560 && countedTrianglesNum == 6101640);
  printf("%s\n", ok ? "PASS" : "FAIL");

  return 0;
}
