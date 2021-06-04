/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/


#include "scan.h"
#include "common.h"

void bScan(queue &q,
           const unsigned int blockSize,
           const unsigned int len,
           buffer<float, 1> &inputBuffer,
           buffer<float, 1> &outputBuffer,
           buffer<float, 1> &blockSumBuffer)
{
  // set the block size
  range<1> gws (len / 2);
  range<1> lws (blockSize / 2);

  q.submit([&] (handler &cgh) {
    auto input = inputBuffer.get_access<sycl_read>(cgh);
    auto output = outputBuffer.get_access<sycl_discard_write>(cgh);
    auto sumBuffer = blockSumBuffer.get_access<sycl_discard_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> block(blockSize, cgh);
    cgh.parallel_for<class lock_scan>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {

      int tid = item.get_local_id(0);
      int gid = item.get_global_id(0);
      int bid = item.get_group(0);
      
      /* Cache the computational window in shared memory */
      block[2*tid]     = input[2*gid];
      block[2*tid + 1] = input[2*gid + 1];
      item.barrier(access::fence_space::local_space);
      
      float cache0 = block[0];
      float cache1 = cache0 + block[1];
      
      /* build the sum in place up the tree */
      for(int stride = 1; stride < blockSize; stride *=2) {
        if(2*tid>=stride) {
          cache0 = block[2*tid-stride]+block[2*tid];
          cache1 = block[2*tid+1-stride]+block[2*tid+1];
        }
        item.barrier(access::fence_space::local_space);
      
        block[2*tid] = cache0;
        block[2*tid+1] = cache1;
      
        item.barrier(access::fence_space::local_space);
      }
      
      /* store the value in sum buffer before making it to 0 */   
      sumBuffer[bid] = block[blockSize-1];
      
      /*write the results back to global memory */
      if(tid==0) {
        output[2*gid]     = 0;
        output[2*gid+1]   = block[2*tid];
      } else {
        output[2*gid]     = block[2*tid-1];
        output[2*gid + 1] = block[2*tid];
      }
    });
  });
}

void pScan(queue &q,
           const unsigned int blockSize,
           const unsigned int len,
           buffer<float, 1> &inputBuffer,
           buffer<float, 1> &outputBuffer)
{
  range<1> gws (len / 2);
  range<1> lws (len / 2);

  q.submit([&] (handler &cgh) {
    auto input = inputBuffer.get_access<sycl_read>(cgh);
    auto output = outputBuffer.get_access<sycl_read_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> block(len+1, cgh);
    cgh.parallel_for<class partial_scan>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {

      int tid = item.get_local_id(0);
      int gid = item.get_global_id(0);
      //int bid = item.get_group(0);

      /* Cache the computational window in shared memory */
      block[2*tid]     = input[2*gid];
      block[2*tid + 1] = input[2*gid + 1];
      item.barrier(access::fence_space::local_space);

      float cache0 = block[0];
      float cache1 = cache0 + block[1];

      /* build the sum in place up the tree */
      for(int stride = 1; stride < blockSize; stride *=2) {

        if(2*tid>=stride) {
          cache0 = block[2*tid-stride]+block[2*tid];
          cache1 = block[2*tid+1-stride]+block[2*tid+1];
        }
        item.barrier(access::fence_space::local_space);

        block[2*tid] = cache0;
        block[2*tid+1] = cache1;

        item.barrier(access::fence_space::local_space);
      }

      /*write the results back to global memory */
      if(tid==0) {
        output[2*gid]     = 0;
        output[2*gid+1]   = block[2*tid];
      } else {
        output[2*gid]     = block[2*tid-1];
        output[2*gid + 1] = block[2*tid];
      }
    });
  });

}

void bAddition(queue &q,
               const unsigned int blockSize,
               const unsigned int len,
               buffer<float, 1> &inputBuffer,
               buffer<float, 1> &outputBuffer)
{
  // set the block size
  range<1> gws (len);
  range<1> lws (blockSize);

  // Enqueue a kernel run call
  q.submit([&] (handler &cgh) {
    auto input = inputBuffer.get_access<sycl_read>(cgh);
    auto output = outputBuffer.get_access<sycl_read_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> value(1, cgh);
    cgh.parallel_for<class block_add>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      int globalId = item.get_global_id(0);
      int groupId = item.get_group(0);
      int localId = item.get_local_id(0);

      /* Only 1 thread of a group will read from global buffer */
      if(localId == 0) value[0] = input[groupId];
      item.barrier(access::fence_space::local_space);

      output[globalId] += value[0];
    });
  });
}


/*
 * Scan for verification
 */
void scanLargeArraysCPUReference(
    float * output,
    float * input,
    const unsigned int length)
{
  output[0] = 0;

  for(unsigned int i = 1; i < length; ++i)
  {
    output[i] = input[i-1] + output[i-1];
  }
}


int main(int argc, char * argv[])
{
  int iterations = atoi(argv[1]);
  int length = atoi(argv[2]);
  int blockSize = atoi(argv[3]);

  if(iterations < 1)
  {
    std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if(!isPowerOf2(length))
  {
    length = roundToPowerOf2(length);
  }

  if((length/blockSize>GROUP_SIZE)&&(((length)&(length-1))!=0))
  {
    std::cout << "Invalid length: " << length << std::endl;
    return -1;
  }

  // input buffer size
  unsigned int sizeBytes = length * sizeof(float);

  float* input = (float*) malloc (sizeBytes);

  // store device results for verification
  float* output = (float*) malloc (sizeBytes);

  // random initialisation of input
  fillRandom<float>(input, length, 1, 0, 255);

  blockSize = (blockSize < length/2) ? blockSize : length/2;

  // Calculate number of passes required
  float t = std::log((float)length) / std::log((float)blockSize);
  unsigned int pass = (unsigned int)t;

  // If t is equal to pass
  if(std::fabs(t - (float)pass) < 1e-7)
  {
    pass--;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // Create input buffer on device
  buffer<float, 1> inputBuffer (input, length);

  // Allocate output buffers
  std::vector<buffer<float,1>> outputBuffer;

  for(unsigned int i = 0; i < pass; i++)
  {
    int size = (int)(length / std::pow((float)blockSize,(float)i));
    outputBuffer.emplace_back(buffer<float, 1>(size));
  }

  // Allocate blockSumBuffers
  std::vector<buffer<float,1>> blockSumBuffer;

  for(unsigned int i = 0; i < pass; i++)
  {
    int size = (int)(length / std::pow((float)blockSize,(float)(i + 1)));
    blockSumBuffer.emplace_back(buffer<float, 1>(size));
  }

  // Create a tempBuffer on device
  int tempLength = (int)(length / std::pow((float)blockSize, (float)pass));

  buffer<float, 1> tempBuffer (tempLength);

  std::cout << "Executing kernel for " <<
    iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" <<
    std::endl;

  for(int n = 0; n < iterations; n++)
  {
    // Do block-wise sum
    bScan(q, blockSize, length, inputBuffer, outputBuffer[0], blockSumBuffer[0]);

    for(int i = 1; i < (int)pass; i++)
    {
      int size = (int)(length / std::pow((float)blockSize,(float)i));
      bScan(q, blockSize, size, blockSumBuffer[i - 1], outputBuffer[i], blockSumBuffer[i]);
    }

    // Do scan to tempBuffer
    pScan(q, blockSize, tempLength, blockSumBuffer[pass - 1], tempBuffer);

    // Do block-addition on outputBuffers
    bAddition(q, blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(pass - 1))),
          tempBuffer, outputBuffer[pass - 1]);

    for(int i = pass - 1; i > 0; i--)
    {
      bAddition(q, blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(i - 1))),
            outputBuffer[i], outputBuffer[i - 1]);
    }
  }

  q.submit([&] (handler &cgh) {
    auto acc = outputBuffer[0].get_access<sycl_read>(cgh);
    cgh.copy(acc, output);
  }).wait();

  // verification
  float* verificationOutput = (float*)malloc(sizeBytes);
  memset(verificationOutput, 0, sizeBytes);

  // reference implementation
  scanLargeArraysCPUReference(verificationOutput, input, length);

  // compare the results and see if they match
  if (compare<float>(output, verificationOutput, length, (float)0.001))
    std::cout << "Passed!\n" << std::endl;
  else
    std::cout << "Failed\n" << std::endl;

  free(input);
  free(output);
  free(verificationOutput);
  return 0;
}
