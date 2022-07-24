#include "tc.h"

#define __device__
#define __global__
#define __syncthreads() item.barrier(access::fence_space::local_space)

template <typename T>
class tc_bs32;

template <typename T>
class tc_bs64;

template <typename T>
class tc_bs96;

template <typename T>
class tc_bs128;

template <typename T>
class tc_bs192;

template <typename T>
class tc_bs256;

template <typename T, unsigned int blockSize, unsigned int dataLength>
__device__ void conditionalWarpReduce(nd_item<1> &item, volatile T *sharedData)
{
  const int threadIdx_x = item.get_local_id(0);
  if(blockSize >= dataLength)
  {
    if(threadIdx_x < (dataLength/2))
    {sharedData[threadIdx_x] += sharedData[threadIdx_x+(dataLength/2)];}
    __syncthreads();
  }
}

template <typename T, unsigned int blockSize>
__device__ void warpReduce(
  nd_item<1> &item,
  T* __restrict outDataPtr,
  volatile T* __restrict sharedData)
{
  conditionalWarpReduce<T, blockSize, 64>(item, sharedData);
  conditionalWarpReduce<T, blockSize, 32>(item, sharedData);
  conditionalWarpReduce<T, blockSize, 16>(item, sharedData);
  conditionalWarpReduce<T, blockSize, 8>(item, sharedData);
  conditionalWarpReduce<T, blockSize, 4>(item, sharedData);
  if(item.get_local_id(0) == 0)
    *outDataPtr = sharedData[0] + sharedData[1];
}

template <typename T, unsigned int blockSize, unsigned int dataLength>
__device__ void conditionalReduce(nd_item<1> &item, volatile T* __restrict sharedData)
{
  const int threadIdx_x = item.get_local_id(0);
  if(blockSize >= dataLength)
  {
    if(threadIdx_x < (dataLength/2))
    {sharedData[threadIdx_x] += sharedData[threadIdx_x+(dataLength/2)];}
    __syncthreads();
  }

  if((blockSize < dataLength) && (blockSize > (dataLength/2)))
  {
    if(threadIdx_x+(dataLength/2) < blockSize)
    {sharedData[threadIdx_x] += sharedData[threadIdx_x+(dataLength/2)];}
    __syncthreads();
  }
}

template <typename T, unsigned int blockSize>
__device__ void blockReduce(
  nd_item<1> &item,
  T* __restrict outGlobalDataPtr,
  volatile T* __restrict sharedData)
{
  __syncthreads();
  conditionalReduce<T, blockSize, 256>(item, sharedData);
  conditionalReduce<T, blockSize, 128>(item, sharedData);

  warpReduce<T, blockSize>(item, outGlobalDataPtr, sharedData);
  __syncthreads();
}

// 
template <typename T>
__device__ void initialize(const T diag_id,
    const T u_len, T v_len,
    T* const __restrict u_min, T* const __restrict u_max,
    T* const __restrict v_min, T* const __restrict v_max,
    T* const __restrict found)
{
  if (diag_id == 0)
  {
    *u_min=*u_max=*v_min=*v_max=0;
    *found=1;
  }
  else if (diag_id < u_len)
  {
    *u_min=0; *u_max=diag_id;
    *v_max=diag_id;*v_min=0;
  }
  else if (diag_id < v_len)
  {
    *u_min=0; *u_max=u_len;
    *v_max=diag_id;*v_min=diag_id-u_len;
  }
  else
  {
    *u_min=diag_id-v_len; *u_max=u_len;
    *v_min=diag_id-u_len; *v_max=v_len;
  }
}

template <typename T>
__device__ void calcWorkPerThread(const T uLength,
    const T vLength, const T threadsPerIntersection,
    const T threadId,
    T * const __restrict outWorkPerThread,
    T * const __restrict outDiagonalId)
{
  T totalWork = uLength + vLength;
  T remainderWork = totalWork%threadsPerIntersection;
  T workPerThread = totalWork/threadsPerIntersection;

  T longDiagonals = (threadId > remainderWork) ? remainderWork:threadId;
  T shortDiagonals = (threadId > remainderWork) ? threadId - remainderWork:0;

  *outDiagonalId = ((workPerThread+1)*longDiagonals) + (workPerThread*shortDiagonals);
  *outWorkPerThread = workPerThread + (threadId < remainderWork);
}

template <typename T>
__device__ void bSearch(
    unsigned int found,
    const T diagonalId,
    T const * const __restrict uNodes,
    T const * const __restrict vNodes,
    T const * const __restrict uLength,
    T * const __restrict outUMin,
    T * const __restrict outUMax,
    T * const __restrict outVMin,
    T * const __restrict outVMax,
    T * const __restrict outUCurr,
    T * const __restrict outVCurr)
{
  T length;
  while(!found)
  {
    *outUCurr = (*outUMin + *outUMax)>>1;
    *outVCurr = diagonalId - *outUCurr;
    if(*outVCurr >= *outVMax)
    {
      length = *outUMax - *outUMin;
      if(length == 1)
      {
        found = 1;
        continue;
      }
    }

    unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr-1];
    unsigned int comp2 = uNodes[*outUCurr-1] > vNodes[*outVCurr];
    if(comp1 && !comp2)
    {
      found = 1;
    }
    else if(comp1)
    {
      *outVMin = *outVCurr;
      *outUMax = *outUCurr;
    }
    else
    {
      *outVMax = *outVCurr;
      *outUMin = *outUCurr;
    }
  }

  if((*outVCurr >= *outVMax) && (length == 1) && (*outVCurr > 0) &&
      (*outUCurr > 0) && (*outUCurr < (*uLength - 1)))
  {
    unsigned int comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
    unsigned int comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
    if(!comp1 && !comp2){(*outUCurr)++; (*outVCurr)--;}
  }
}

template <typename T>
__device__ T fixThreadWorkEdges(const T uLength, const T vLength,
    T * const __restrict uCurr, T * const __restrict vCurr,
    T const * const __restrict uNodes, T const * const __restrict vNodes)
{
  unsigned int uBigger = (*uCurr > 0) && (*vCurr < vLength) &&
    (uNodes[*uCurr-1] == vNodes[*vCurr]);
  unsigned int vBigger = (*vCurr > 0) && (*uCurr < uLength) &&
    (vNodes[*vCurr-1] == uNodes[*uCurr]);
  *uCurr += vBigger;
  *vCurr += uBigger;

  return (uBigger + vBigger);
}

template <typename T>
__device__ void intersectCount(const T uLength, const T vLength,
    T const * const __restrict uNodes, T const * const __restrict vNodes,
    T * const __restrict uCurr, T * const __restrict vCurr,
    T * const __restrict workIndex, T * const __restrict workPerThread,
    T * const __restrict triangles, T found)
{
  if((*uCurr < uLength) && (*vCurr < vLength))
  {
    T comp;
    while(*workIndex < *workPerThread)
    {
      comp = uNodes[*uCurr] - vNodes[*vCurr];
      *triangles += (comp == 0);
      *uCurr += (comp <= 0);
      *vCurr += (comp >= 0);
      *workIndex += (comp == 0) + 1;

      if((*vCurr == vLength) || (*uCurr == uLength))
      {
        break;
      }
    }
    *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
  }
}

// u_len < v_len
template <typename T>
__device__ T count_triangles(T u, T const * const __restrict u_nodes, T u_len,
    T v, T const * const __restrict v_nodes, T v_len, T threads_per_block,
    volatile T* __restrict firstFound, T tId)
{
  // Partitioning the work to the multiple thread of a single GPU processor. 
  // The threads should get a near equal number of the elements to Tersect - this number will be off by 1.
  T work_per_thread, diag_id;
  calcWorkPerThread(u_len, v_len, threads_per_block, tId, &work_per_thread, &diag_id);
  T triangles = 0;
  T work_index = 0,found=0;
  T u_min,u_max,v_min,v_max,u_curr,v_curr;

  firstFound[tId]=0;

  if(work_per_thread>0)
  {
    // For the binary search, we are figuring out the initial poT of search.
    initialize(diag_id, u_len, v_len,&u_min, &u_max,&v_min, &v_max,&found);
    u_curr = 0; v_curr = 0;

    bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max, &v_min,
        &v_max, &u_curr, &v_curr);

    T sum = fixThreadWorkEdges(u_len, v_len, &u_curr, &v_curr, u_nodes, v_nodes);
    work_index += sum;
    if(tId > 0) firstFound[tId-1] = sum;
    triangles += sum;
    intersectCount(u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
        &work_index, &work_per_thread, &triangles, firstFound[tId]);
  }
  return triangles;
}

template <typename T>
__device__ void calcWorkPerBlock(
    nd_item<1> &item,
    const T numVertices,
    T * const __restrict outMpStart,
    T * const __restrict outMpEnd)
{
  const int gridDim_x = item.get_group_range(0);
  const int blockIdx_x = item.get_group(0);

  T verticesPerMp = numVertices/gridDim_x;
  T remainderBlocks = numVertices % gridDim_x;
  T extraVertexBlocks = (blockIdx_x > remainderBlocks)?
    remainderBlocks:blockIdx_x;
  T regularVertexBlocks = (blockIdx_x > remainderBlocks)?
    blockIdx_x - remainderBlocks:0;

  T mpStart = ((verticesPerMp+1)*extraVertexBlocks)
    + (verticesPerMp*regularVertexBlocks);
  *outMpStart = mpStart;
  *outMpEnd = mpStart + verticesPerMp + (blockIdx_x < remainderBlocks);
}

template <typename T, unsigned int blockSize>
__global__ void count_all_trianglesGPU (
    nd_item<1> &item,
    T * const __restrict s_triangles, // shared
    T * const __restrict firstFound,  // shared
    const T nv,
    T const * const __restrict d_off,
    T const * const __restrict d_ind,
    T * const __restrict outPutTriangles,
    const T threads_per_block,
    const T number_blocks, const T shifter)
{
  T tx = item.get_local_id(0);
  T this_mp_start, this_mp_stop;
  calcWorkPerBlock(item, nv, &this_mp_start, &this_mp_stop);

  T adj_offset=tx>>shifter;
  T* firstFoundPos=firstFound + (adj_offset<<shifter);
  for (T src = this_mp_start; src < this_mp_stop; src++)
  {
    T srcLen=d_off[src+1]-d_off[src];
    T tCount = 0;
    for(T iter=d_off[src]+adj_offset; iter<d_off[src+1]; iter+=number_blocks)
    {
      T dest = d_ind[iter];
      T destLen = d_off[dest+1]-d_off[dest];
      bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
      if(avoidCalc) continue;

      bool sourceSmaller = (srcLen<destLen);
      T small = sourceSmaller? src : dest;
      T large = sourceSmaller? dest : src;
      T small_len = sourceSmaller? srcLen : destLen;
      T large_len = sourceSmaller? destLen : srcLen;

      T const * const small_ptr = d_ind + d_off[small];
      T const * const large_ptr = d_ind + d_off[large];
      tCount += count_triangles(
          small, small_ptr, small_len,
          large, large_ptr, large_len,
          threads_per_block, firstFoundPos, tx%threads_per_block);
    }
    s_triangles[tx] = tCount;
    blockReduce<T, blockSize>(item, &outPutTriangles[src],s_triangles);
  }
}

#define CALL_SYCL_KERNEL(BS) \
      q.submit([&] (handler &cgh) { \
        auto off = d_off.template get_access<sycl_read>(cgh); \
        auto ind = d_ind.template get_access<sycl_read>(cgh); \
        auto out = d_out.template get_access<sycl_write>(cgh); \
        accessor<T, 1, sycl_read_write, access::target::local> triangle(BS, cgh); \
        accessor<T, 1, sycl_read_write, access::target::local> firstfound(BS, cgh); \
        cgh.parallel_for<class tc_bs##BS<T>>(nd_range<1>(range<1>(BS*numberBlocks), range<1>(BS)), \
                                       [=] (nd_item<1> item) { \
          count_all_trianglesGPU<T, BS> (\
            item, triangle.get_pointer(), firstfound.get_pointer(),\
            nv, off.get_pointer(), ind.get_pointer(), out.get_pointer(),\
            threads_per_block, number_blocks, shifter);\
        });\
      });

// call triangle count kernel
template <typename T>
void kernelCall(
    queue &q,
    unsigned int numberBlocks,
    unsigned int numberThreads,
    const T nv,
    buffer<const T, 1> &d_off,
    buffer<const T, 1> &d_ind,
    buffer<T, 1> &d_out,
    const T threads_per_block,
    const T number_blocks,
    const T shifter)
{
  switch (numberThreads) {
    case 32: CALL_SYCL_KERNEL(32) break;
    case 64: CALL_SYCL_KERNEL(64) break;
    case 96: CALL_SYCL_KERNEL(96) break;
    case 128: CALL_SYCL_KERNEL(128) break;
    case 192: CALL_SYCL_KERNEL(192) break;
    case 256: CALL_SYCL_KERNEL(256) break;
    default: ;
  }
}

template <typename T>
void allParamTestGPURun(queue &q, Param param)
{
  T* offsetVector;
  T* indexVector;
  T vertexCount;
  T edgeCount;

  bool ok = readGraph<T>(param.fileName, offsetVector, indexVector, vertexCount, edgeCount);
  if (!ok) return;

  q.wait();
  auto memAllocStart = std::chrono::system_clock::now();

  buffer<const T, 1> dOffset (offsetVector, vertexCount + 1);
  buffer<const T, 1> dIndex (indexVector, edgeCount);

  T *triangle = new T[vertexCount];

  buffer<T, 1> dTriangle (vertexCount);

  auto memAllocEnd = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> memAllocDuration = memAllocEnd - memAllocStart;

  // output file name 
  std::string separator = std::string(".o.");
  std::string fileOutName = param.fileName + separator + std::to_string(param.blocks);
  std::ofstream writeFile(fileOutName);

  writeFile<<"paramBlockSize"<<"\t"
           <<"paramThreadsPerIntsctn"<<"\t"
           <<"memAllocDuration(ms)"<<"\t"
           <<"execDuration(ms)"<<"\t"
           <<"kernelDuration(ms)"<<"\t"
           <<"execDuration+memAllocDuration(ms)"<<"\t"
           <<"sumTriangles"<<"\n";

  for(auto paramBlockSize : globalParam::blockSizeParam)
  {
    for(auto paramThreadsPerIntsctn : globalParam::threadPerIntersectionParam)
    {
      q.submit([&] (handler &cgh) {
        auto acc = dTriangle.template get_access<sycl_write>(cgh);
        cgh.fill(acc, (T)0);
      }).wait();
      
      // timing data transfer and kernel execution on a device
      auto execStart = std::chrono::system_clock::now();
      unsigned int blocks = param.blocks;
      unsigned int blockSize = paramBlockSize;
      T threadsPerIntsctn = paramThreadsPerIntsctn;
      T intsctnPerBlock = paramBlockSize/paramThreadsPerIntsctn;
      T threadShift = std::log2(paramThreadsPerIntsctn);

      // timing kernel execution on a device
      auto krnlStart = std::chrono::system_clock::now();
      kernelCall<T>(q, blocks, blockSize, vertexCount, dOffset,
          dIndex, dTriangle, threadsPerIntsctn, intsctnPerBlock, threadShift);
      q.wait();
      auto krnlEnd = std::chrono::system_clock::now();
      std::chrono::duration<float, std::milli> krnlDuration = krnlEnd - krnlStart;

      q.submit([&] (handler &cgh) {
        auto acc = dTriangle.template get_access<sycl_read>(cgh);
        cgh.copy(acc, triangle);
      }).wait();

      auto execEnd = std::chrono::system_clock::now();
      std::chrono::duration<float, std::milli> execDuration = execEnd - execStart;
      
      T sumTriangles = 0;
      for (int i = 0; i < vertexCount; i++)
        sumTriangles += triangle[i];

      writeFile<<paramBlockSize<<"\t"
               <<paramThreadsPerIntsctn<<"\t"
               <<memAllocDuration.count()<<"\t"
               <<execDuration.count()<<"\t"
               <<krnlDuration.count()<<"\t"
               <<execDuration.count()+memAllocDuration.count()<<"\t"
               <<sumTriangles<<"\n";
    }
  }
  writeFile.close();
    
  delete[] offsetVector;
  delete[] indexVector;
  delete[] triangle;
}

template void allParamTestGPURun<int>(queue &q, Param param);
template void allParamTestGPURun<long>(queue &q, Param param);
