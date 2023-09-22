/*------------------------------------------------------------------------------
* Copyright 2015-16: Tom Deakin, Simon McIntosh-Smith, University of Bristol HPC
* Based on John D. McCalpin’s original STREAM benchmark for CPUs
*------------------------------------------------------------------------------
* License:
*  1. You are free to use this program and/or to redistribute
*     this program.
*  2. You are free to modify this program for your own use,
*     including commercial use, subject to the publication
*     restrictions in item 3.
*  3. You are free to publish results obtained from running this
*     program, or from works that you derive from this program,
*     with the following limitations:
*     3a. In order to be referred to as "BabelStream benchmark results",
*         published results must be in conformance to the BabelStream
*         Run Rules published at
*         http://github.com/UoB-HPC/BabelStream/wiki/Run-Rules
*         and incorporated herein by reference.
*         The copyright holders retain the
*         right to determine conformity with the Run Rules.
*     3b. Results based on modified source code or on runs not in
*         accordance with the BabelStream Run Rules must be clearly
*         labelled whenever they are published.  Examples of
*         proper labelling include:
*         "tuned BabelStream benchmark results"
*         "based on a variant of the BabelStream benchmark code"
*         Other comparable, clear and reasonable labelling is
*         acceptable.
*     3c. Submission of results to the BabelStream benchmark web site
*         is encouraged, but not required.
*  4. Use of this program or creation of derived works based on this
*     program constitutes acceptance of these licensing restrictions.
*  5. Absolutely no warranty is expressed or implied.
*/

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sycl/sycl.hpp>


// Thread block size
#define TBSIZE 256

// Number of thread blocks for the DOT kernel 
#define DOT_NUM_BLOCKS 256

// Scalar constanst for the mul, triad and nstream kernels
#define SCALAR (0.4)

// Default size of 2^25
int ARRAY_SIZE = 33554432;

// Kernel execution times
unsigned int num_times = 100;

// Forward declarations
// SYCL spec: If the lambda function relies on template arguments,
// then if specified, the name of the lambda function must contain 
// those template arguments which must also be forward declarable at namespace scope.
template <typename T>
class init_kernel;

template <typename T>
class copy_kernel;

template <typename T>
class mul_kernel;

template <typename T>
class add_kernel;

template <typename T>
class triad_kernel;

template <typename T>
class nstream_kernel;

template <typename T>
class dot_kernel;

// Initialize buffers da, db, dc with initA, initB, and initC, respectively 
template <typename T>
void init_arrays(sycl::queue &q, 
                 T *da, T *db, T *dc, 
                 T initA, T initB, T initC)
{
  const int array_size = ARRAY_SIZE; 
  sycl::range<1> gws (array_size);
  sycl::range<1> lws (TBSIZE);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class init_kernel<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const int i = item.get_global_id(0);
      da[i] = initA;
      db[i] = initB;
      dc[i] = initC;
    });
  }).wait();
}


// dc[i] = da[i] for each i
template <typename T>
void copy(sycl::queue &q, T *da, T *dc)
{
  const int array_size = ARRAY_SIZE;
  sycl::range<1> gws (array_size);
  sycl::range<1> lws (TBSIZE);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class copy_kernel<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const int i = item.get_global_id(0);
      dc[i] = da[i];
    });
  });
  q.wait();
}

// db[i] = scalar * dc[i] for each i
template <typename T>
void mul(sycl::queue &q, T *db, T *dc)
{
  const int array_size = ARRAY_SIZE;
  sycl::range<1> gws (array_size);
  sycl::range<1> lws (TBSIZE);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class mul_kernel<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const T scalar = SCALAR;
      const int i = item.get_global_id(0);
      db[i] = scalar * dc[i];
    });
  });
  q.wait();
}

// dc[i] = da[i] + db[i] for each i
template <typename T>
void add(sycl::queue &q, T *da, T *db, T *dc)
{
  const int array_size = ARRAY_SIZE;
  sycl::range<1> gws (array_size);
  sycl::range<1> lws (TBSIZE);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class add_kernel<T>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const int i = item.get_global_id(0);
      dc[i] = da[i] + db[i];
    });
  }).wait();
}


// da[i] = db[i] + scalar * dc[i] for each i
template <typename T>
void triad(sycl::queue &q, T *da, T *db, T *dc)
{
  const int array_size = ARRAY_SIZE;
  sycl::range<1> gws (array_size);
  sycl::range<1> lws (TBSIZE);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class triad_kernel<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const T scalar = SCALAR;
      const int i = item.get_global_id(0);
      da[i] = db[i] + scalar * dc[i];
    });
  }).wait();
}


// da[i] += db[i] + scalar * dc[i] for each i
template <typename T>
void nstream(sycl::queue &q, T *da, T *db, T *dc)
{
  const int array_size = ARRAY_SIZE;
  sycl::range<1> gws (array_size);
  sycl::range<1> lws (TBSIZE);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class nstream_kernel<T>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const T scalar = SCALAR;
      const int i = item.get_global_id(0);
      da[i] += db[i] + scalar * dc[i];
    });
  }).wait();
}

// sum += da[i] * db[i] for each i
template <typename T>
T dot(sycl::queue &q, T *da, T *db, T *dsum, T *sums)
{
  const int array_size = ARRAY_SIZE;
  sycl::range<1> gws (DOT_NUM_BLOCKS * TBSIZE);
  sycl::range<1> lws (TBSIZE);
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<T, 1> tb_sum(sycl::range<1>(TBSIZE), cgh);
    cgh.parallel_for<class dot_kernel<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const size_t lid = item.get_local_id(0);
      const int blockIdx = item.get_group(0);
      const int blockDim = item.get_local_range(0);
      const int gridDim = item.get_group_range(0);

      tb_sum[lid] = 0.0;
      for (int i = item.get_global_id(0);
           i < array_size; i += blockDim * gridDim)
        tb_sum[lid] += da[i] * db[i];

      for (int offset = blockDim / 2; offset > 0; offset /= 2)
      {
        item.barrier(sycl::access::fence_space::local_space);
        if (lid < offset)
        {
          tb_sum[lid] += tb_sum[lid+offset];
        }
      }

      if (lid == 0)
        dsum[blockIdx] = tb_sum[lid];
    });
  });

  // sum up partial sums on a host
  q.memcpy(sums, dsum, DOT_NUM_BLOCKS * sizeof(T)).wait();  

  T sum = 0.0;
  for (int i = 0; i < DOT_NUM_BLOCKS; i++)
    sum += sums[i];
  return sum;
}


// Runs the kernel(s) and prints output.
template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  std::cout << "Running kernels " << num_times << " times" << std::endl;

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T *da = sycl::malloc_device<T>(ARRAY_SIZE, q);
  T *db = sycl::malloc_device<T>(ARRAY_SIZE, q);
  T *dc = sycl::malloc_device<T>(ARRAY_SIZE, q);
  T *dsum = sycl::malloc_device<T>(DOT_NUM_BLOCKS, q);

  // Allocate the host array for partial sums for the dot kernel
  T *sums = (T*)malloc(sizeof(T) * DOT_NUM_BLOCKS);

  if (sizeof(T) == sizeof(float))
    std::cout << "Precision: float" << std::endl;
  else
    std::cout << "Precision: double" << std::endl;

  // MB = 10^6
  std::cout << std::setprecision(1) << std::fixed
    << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout.precision(ss);

  // Initialize device arrays
  init_arrays(q, da, db, dc, (T)0.1, (T)0.2, T(0.0));

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Copy
    t1 = std::chrono::high_resolution_clock::now();
    copy(q, da, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Mul
    t1 = std::chrono::high_resolution_clock::now();
    mul(q, db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Add
    t1 = std::chrono::high_resolution_clock::now();
    add(q, da, db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Triad
    t1 = std::chrono::high_resolution_clock::now();
    triad(q, da, db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Dot
    t1 = std::chrono::high_resolution_clock::now();
    dot(q, da, db, dsum, sums);
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute NStream
    t1 = std::chrono::high_resolution_clock::now();
    nstream(q, da, db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
  }

  // Display timing results
  std::cout
    << std::left << std::setw(12) << "Function"
    << std::left << std::setw(12) << "MBytes/sec"
    << std::left << std::setw(12) << "Min (sec)"
    << std::left << std::setw(12) << "Max"
    << std::left << std::setw(12) << "Average"
    << std::endl
    << std::fixed;

  std::vector<std::string> labels;
  std::vector<size_t> sizes;

  labels = {"Copy", "Mul", "Add", "Triad", "Dot", "Nstream"};
  sizes = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    4 * sizeof(T) * ARRAY_SIZE};

  for (size_t i = 0; i < timings.size(); ++i)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

    double bandwidth = 1.0E-6 * sizes[i] / (*minmax.first);

    std::cout
      << std::left << std::setw(12) << labels[i]
      << std::left << std::setw(12) << std::setprecision(3) << bandwidth
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
      << std::left << std::setw(12) << std::setprecision(5) << average
      << std::endl;
  }
  // Add a blank line
  std::cout << std::endl;

  free(da, q);
  free(db, q);
  free(dc, q);
  free(dsum, q);
  free(sums);
}


int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int parseInt(const char *str, int *output)
{
  char *next;
  *output = strtol(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--arraysize").compare(argv[i]) ||
        !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseInt(argv[i], &ARRAY_SIZE) || ARRAY_SIZE <= 0)
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
        !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--help").compare(argv[i]) ||
        !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
        << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char *argv[])
{
  parseArguments(argc, argv);
  run<float>();
  run<double>();
}


