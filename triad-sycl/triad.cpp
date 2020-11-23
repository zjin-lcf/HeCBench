#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "OptionParser.h"
#include "Timer.h"
#include "Utility.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
  ;
}
// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Implements the Stream Triad benchmark in CUDA.  This benchmark
//   is designed to test CUDA's overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{
  const bool verbose = op.getOptionBool("verbose");
  const int n_passes = op.getOptionInt("passes");

  const int nSizes = 9;
  const size_t blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
  const size_t memSize = 16384;
  const size_t numMaxFloats = 1024 * memSize / 4;
  const size_t halfNumFloats = numMaxFloats / 2;

  // Create some host memory pattern
  srand48(8650341L);
  float *h_mem = (float*) malloc (sizeof(float) * numMaxFloats);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  cl::sycl::queue q(dev_sel);

  // Allocate device memory of maximum sizes
  buffer<float,1> d_memA0 (blockSizes[nSizes - 1] * 1024);
  buffer<float,1> d_memB0 (blockSizes[nSizes - 1] * 1024);
  buffer<float,1> d_memC0 (blockSizes[nSizes - 1] * 1024);
  buffer<float,1> d_memA1 (blockSizes[nSizes - 1] * 1024);
  buffer<float,1> d_memB1 (blockSizes[nSizes - 1] * 1024);
  buffer<float,1> d_memC1 (blockSizes[nSizes - 1] * 1024);

  const float scalar = 1.75f;
  const size_t blockSize = 128;

  // Number of passes. Use a large number for stress testing.
  // A small value is sufficient for computing sustained performance.
  for (int pass = 0; pass < n_passes; ++pass)
  {
    // Step through sizes forward
    for (int i = 0; i < nSizes; ++i)
    {
      int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
      for (int j = 0; j < halfNumFloats; ++j)
        h_mem[j] = h_mem[halfNumFloats + j] = (float) (drand48() * 10.0);

      // Copy input memory to the device
      if (verbose) {
        std::cout << ">> Executing Triad with vectors of length "
          << numMaxFloats << " and block size of "
          << elemsInBlock << " elements." << "\n";
      //sprintf(sizeStr, "Block:%05ldKB", blockSizes[i]);
        printf("Block:%05ldKB\n", blockSizes[i]);
      }

      // start submitting blocks of data of size elemsInBlock
      // overlap the computation of one block with the data
      // download for the next block and the results upload for
      // the previous block
      int crtIdx = 0;
      size_t globalWorkSize = elemsInBlock;

      int TH = Timer::Start();

      q.submit([&] (handler &cgh) {
          auto d_memA0_acc = d_memA0.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
          cgh.copy(h_mem, d_memA0_acc);
          });
      q.submit([&] (handler &cgh) {
          auto d_memB0_acc = d_memB0.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
          cgh.copy(h_mem, d_memB0_acc);
          });
      q.submit([&] (handler &cgh) {
          auto A = d_memA0.get_access<sycl_read>(cgh, range<1>(elemsInBlock));
          auto B = d_memB0.get_access<sycl_read>(cgh, range<1>(elemsInBlock));
          auto C = d_memC0.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
          cgh.parallel_for<class triad_start>(nd_range<1>(range<1>(globalWorkSize), range<1>(blockSize)), [=] (nd_item<1> item) {
              int gid = item.get_global_id(0); 
              C[gid] = A[gid] + scalar*B[gid];
              });
          });


      if (elemsInBlock < numMaxFloats)
      {
        // start downloading data for next block
        q.submit([&] (handler &cgh) {
            auto d_memA1_acc = d_memA1.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
            cgh.copy(h_mem+elemsInBlock, d_memA1_acc);
            });
        q.submit([&] (handler &cgh) {
            auto d_memB1_acc = d_memB1.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
            cgh.copy(h_mem+elemsInBlock, d_memB1_acc);
            });
      }

      int blockIdx = 1;
      unsigned int currStream = 1;
      while (crtIdx < numMaxFloats)
      {
        currStream = blockIdx & 1;
        // Start copying back the answer from the last kernel
        if (currStream)
        {
          q.submit([&] (handler &cgh) {
              auto d_memC0_acc = d_memC0.get_access<sycl_read>(cgh, range<1>(elemsInBlock));
              cgh.copy(d_memC0_acc, h_mem+crtIdx);
              });
        }
        else
        {
          q.submit([&] (handler &cgh) {
              auto d_memC1_acc = d_memC1.get_access<sycl_read>(cgh, range<1>(elemsInBlock));
              cgh.copy(d_memC1_acc, h_mem+crtIdx);
              });
        }

        crtIdx += elemsInBlock;

        if (crtIdx < numMaxFloats)
        {
          // Execute the kernel
          if (currStream)
          {
            q.submit([&] (handler &cgh) {
                auto A = d_memA1.get_access<sycl_read>(cgh);
                auto B = d_memB1.get_access<sycl_read>(cgh);
                auto C = d_memC1.get_access<sycl_write>(cgh);
                cgh.parallel_for<class triad_curr>(nd_range<1>(range<1>(globalWorkSize), range<1>(blockSize)), [=] (nd_item<1> item) {
                    int gid = item.get_global_id(0); 
                    C[gid] = A[gid] + scalar*B[gid];
                    });
                });
          }
          else
          {
            q.submit([&] (handler &cgh) {
                auto A = d_memA0.get_access<sycl_read>(cgh);
                auto B = d_memB0.get_access<sycl_read>(cgh);
                auto C = d_memC0.get_access<sycl_write>(cgh);
                cgh.parallel_for<class triad_next>(nd_range<1>(range<1>(globalWorkSize), range<1>(blockSize)), [=] (nd_item<1> item) {
                    int gid = item.get_global_id(0); 
                    C[gid] = A[gid] + scalar*B[gid];
                    });
                });
          }
        }

        if (crtIdx+elemsInBlock < numMaxFloats)
        {
          // Download data for next block
          if (currStream)
          {
            q.submit([&] (handler &cgh) {
                auto d_memA0_acc = d_memA0.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
                cgh.copy(h_mem+crtIdx+elemsInBlock, d_memA0_acc);
                });
            q.submit([&] (handler &cgh) {
                auto d_memB0_acc = d_memB0.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
                cgh.copy(h_mem+crtIdx+elemsInBlock, d_memB0_acc);
                });
          }
          else
          {
            q.submit([&] (handler &cgh) {
                auto d_memA1_acc = d_memA1.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
                cgh.copy(h_mem+crtIdx+elemsInBlock, d_memA1_acc);
                });
            q.submit([&] (handler &cgh) {
                auto d_memB1_acc = d_memB1.get_access<sycl_write>(cgh, range<1>(elemsInBlock));
                cgh.copy(h_mem+crtIdx+elemsInBlock, d_memB1_acc);
                });
          }
        }
        blockIdx += 1;
        currStream = !currStream;
      }
      q.wait();
      double time = Timer::Stop(TH, "thread synchronize");

      double triad = ((double)numMaxFloats * 2.0) / (time*1e9);
      if (verbose)
        std::cout << "TriadFlops " << triad << " GFLOPS/s\n";

      //resultDB.AddResult("TriadFlops", sizeStr, "GFLOP/s", triad);

      double bdwth = ((double)numMaxFloats*sizeof(float)*3.0)
        / (time*1000.*1000.*1000.);
      //resultDB.AddResult("TriadBdwth", sizeStr, "GB/s", bdwth);
      if (verbose)
        std::cout << "TriadBdwth " << bdwth << " GB/s\n";

      // Checking memory for correctness. The two halves of the array
      // should have the same results.
      if (verbose) std::cout << ">> checking memory\n";
      for (int j=0; j<halfNumFloats; ++j)
      {
        if (h_mem[j] != h_mem[j+halfNumFloats])
        {
          std::cout << "Error; hostMem[" << j << "]=" << h_mem[j]
            << " is different from its twin element hostMem["
            << (j+halfNumFloats) << "]: "
            << h_mem[j+halfNumFloats] << "stopping check\n";
          break;
        }
      }
      if (verbose) std::cout << ">> finish!" << std::endl;

      // Zero out the test host memory
      for (int j=0; j<numMaxFloats; ++j)
        h_mem[j] = 0.0f;
    }
  }

  // Cleanup
  free(h_mem);
}
