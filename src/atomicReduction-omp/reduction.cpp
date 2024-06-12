/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>
#include <cmath>

int main(int argc, char** argv)
{
  int arrayLength = 52428800;
  int threads=256;
  int N = 32;

  if (argc == 4) {
    arrayLength=atoi(argv[1]);
    threads=atoi(argv[2]);
    N=atoi(argv[3]);
  }

  std::cout << "Array size: " << arrayLength*sizeof(int)/1024.0/1024.0 << " MB"<<std::endl;
  std::cout << "Thread block size: " << threads << std::endl;
  std::cout << "Repeat the kernel execution: " << N << " times" << std::endl;

  int* array=(int*)malloc(arrayLength*sizeof(int));
  int checksum =0;
  for(int i=0;i<arrayLength;i++) {
    array[i]=rand()%2;
    checksum+=array[i];
  }

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  int sum;

  #pragma omp target data map(to: array[0:arrayLength]) map(alloc: sum)
  {
    int blocks=std::min((arrayLength+threads-1)/threads,2048);
    // warmup
    for(int n=0;n<N;n++) {
      sum = 0;
      #pragma omp target update to(sum)
      #pragma omp target teams distribute parallel for \
      num_teams(blocks) thread_limit(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength; i++) {
        sum += array[i];
      }
    }

    // start timing
    t1 = std::chrono::high_resolution_clock::now();
    for(int n=0;n<N;n++) {
      sum = 0;
      #pragma omp target update to(sum)
      #pragma omp target teams distribute parallel for \
      num_teams(blocks) thread_limit(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength; i++) {
        sum += array[i];
      }
    }
    #pragma omp target update from(sum)
    t2 = std::chrono::high_resolution_clock::now();
    double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    float GB=(float)arrayLength*sizeof(int)*N;
    std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;


    printf("%d %d\n", sum, checksum);
    if(sum==checksum)
      std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else {
      std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;
      exit(1);
    }

    t1 = std::chrono::high_resolution_clock::now();
    for(int n=0;n<N;n++) {
      sum = 0;
      #pragma omp target update to(sum)
      #pragma omp target teams distribute parallel for \
      num_teams(blocks/2) thread_limit(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength; i=i+2) { 
        sum += array[i] + array[i+1];
      }
    }
    #pragma omp target update from(sum)
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    if(sum==checksum)
      std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
      std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(int n=0;n<N;n++) {
      sum = 0;
      #pragma omp target update to(sum)
      #pragma omp target teams distribute parallel for \
      num_teams(blocks/4) thread_limit(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength; i=i+4) { 
        sum += array[i] + array[i+1] + array[i+2] + array[i+3];
      }
    }
    #pragma omp target update from(sum)
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    if(sum==checksum)
      std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
      std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(int n=0;n<N;n++) {
      sum = 0;
      #pragma omp target update to(sum)
      #pragma omp target teams distribute parallel for \
      num_teams(blocks/8) thread_limit(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength; i=i+8) { 
        sum += array[i] + array[i+1] + array[i+2] + array[i+3] + 
               array[i+4] + array[i+5] + array[i+6] + array[i+7];
      }
    }
    #pragma omp target update from(sum)
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    if(sum==checksum)
      std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
      std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(int n=0;n<N;n++) {
      sum = 0;
      #pragma omp target update to(sum)
      #pragma omp target teams distribute parallel for \
      num_teams(blocks/16) thread_limit(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength; i=i+16) { 
        sum += array[i] + array[i+1] + array[i+2] + array[i+3] + 
               array[i+4] + array[i+5] + array[i+6] + array[i+7] +
               array[i+8] + array[i+9] + array[i+10] + array[i+11] +
               array[i+12] + array[i+13] + array[i+14] + array[i+15];
      }
    }
    #pragma omp target update from(sum)
    t2 = std::chrono::high_resolution_clock::now();
    times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    GB=(float)arrayLength*sizeof(int)*N;
    std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

    if(sum==checksum)
      std::cout<<"VERIFICATION: result is CORRECT"<<std::endl<<std::endl;
    else
      std::cout<<"VERIFICATION: result is INCORRECT!!"<<std::endl<<std::endl;
  }
  free(array);
  return 0;
}
