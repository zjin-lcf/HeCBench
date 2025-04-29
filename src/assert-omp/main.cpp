/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <exception>
#include <omp.h>

// Tests assert function.
// Thread whose id > N will print assertion failed error message.
void testKernel(const int numTeams, const int numThreads, int N)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (int gid = 0; gid < N; gid++)
    assert(gid < N);
}

// Performance impact of assert()
void perfKernel(const int numTeams, const int numThreads)
{
  #pragma omp target teams num_teams(numTeams)
  {
    #pragma omp parallel num_threads(numThreads)
    {
      int gid = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();
      assert(gid <= omp_get_num_threads() * omp_get_num_teams());
      int s = 0;
      for (int n = 1; n <= gid; n++) {
        s++; assert(s <= gid);
      }
    }
  }
}

void perfKernel2(const int numTeams, const int numThreads)
{
  #pragma omp target teams num_teams(numTeams)
  {
    #pragma omp parallel num_threads(numThreads)
    {
      int gid = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();
      int s = 0;
      for (int n = 1; n <= gid; n++) {
        s++; assert(s <= gid);
      }
    }
  }
}

// Declaration, forward
bool runPerf(int argc, char **argv);
bool runTest(int argc, char **argv);

int main(int argc, char **argv)
{
  // The test expects no assertError
  runPerf(argc, argv);

  // The test expects assertError
  bool testResult = runTest(argc, argv);

  printf("Test assert completed, returned %s\n",
         testResult ? "OK" : "ERROR!");

  if (!testResult) return EXIT_FAILURE;

  exit(EXIT_SUCCESS);
}

bool runTest(int argc, char **argv) {
  int Nblocks = 2;
  int Nthreads = 32;

  printf("\nLaunch kernel to generate assertion failures\n");

  try {
    // Synchronize (flushes assert output).
    printf("\n-- Begin assert output\n\n");
    testKernel(Nblocks, Nthreads, 60);
    printf("\n-- End assert output\n\n");
  }
  catch (...) {}
  return true;
}

bool runPerf(int argc, char **argv)
{
  int Nblocks = 1000;
  int Nthreads = 256;

  printf("\nLaunch kernel to evaluate the impact of assertion on performance \n");

  printf("Each thread in the kernel executes threadID + 1 assertions\n");
  auto start = std::chrono::steady_clock::now();
  perfKernel(Nblocks, Nthreads);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Kernel time : %f\n", time.count());

  printf("Each thread in the kernel executes threadID assertions\n");
  perfKernel2(Nblocks, Nthreads);
  end = std::chrono::steady_clock::now();
  time = end - start;
  printf("Kernel time : %f\n", time.count());

  return true;
}
