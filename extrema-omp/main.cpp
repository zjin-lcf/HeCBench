// Copyright (c) 2019-2020, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#pragma omp declare target
inline void clip_plus( const bool &clip, const int &n, int &plus ) {
  if ( clip ) {
    if ( plus >= n ) {
      plus = n - 1;
    }
  } else {
    if ( plus >= n ) {
      plus -= n;
    }
  }
}
#pragma omp end declare target

#pragma omp declare target
inline void clip_minus( const bool &clip, const int &n, int &minus ) {
  if ( clip ) {
    if ( minus < 0 ) {
      minus = 0;
    }
  } else {
    if ( minus < 0 ) {
      minus += n;
    }
  }
}
#pragma omp end declare target

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 1D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
void cpu_relextrema_1D(
  const int  n,
  const int  order,
  const bool clip,
  const T * inp,
  bool * results)
{
  for ( int tid = 0; tid < n; tid++ ) {

    const T data = inp[tid];
    bool    temp = true;

    for ( int o = 1; o < ( order + 1 ); o++ ) {
      int plus = tid + o;
      int minus = tid - o;

      clip_plus( clip, n, plus );
      clip_minus( clip, n, minus );

      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp;
  }
}

template<typename T>
void cpu_relextrema_2D(
  const int  in_x,
  const int  in_y,
  const int  order,
  const bool clip,
  const int  axis,
  const T * inp,
  bool * results) 
{
  for (int tx = 0; tx < in_y; tx++)
    for (int ty = 0; ty < in_x; ty++) {

      int tid = tx * in_x + ty ;

      const T data = inp[tid] ;
      bool    temp = true ;

      for ( int o = 1; o < ( order + 1 ); o++ ) {

        int plus;
        int minus;

        if ( axis == 0 ) {
          plus  = tx + o;
          minus = tx - o;

          clip_plus( clip, in_y, plus );
          clip_minus( clip, in_y, minus );

          plus  = plus * in_x + ty;
          minus = minus * in_x + ty;
        } else {
          plus  = ty + o;
          minus = ty - o;

          clip_plus( clip, in_x, plus );
          clip_minus( clip, in_x, minus );

          plus  = tx * in_x + plus;
          minus = tx * in_x + minus;
        }

        temp &= data > inp[plus] ;
        temp &= data >= inp[minus] ;
      }
      results[tid] = temp;
    }
}

template <typename T>
long test_1D (const int length, const int order, const bool clip,
              const int repeat, const char* type) 
{
  T* inp = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    inp[i] = rand() % length;

  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);
  
  long time;
  #pragma omp target data map(to: inp[0:length]) map(from: gpu_r[0:length])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int tid = 0; tid < length; tid++) {
        const T data = inp[tid];
        bool    temp = true;

        for ( int o = 1; o < ( order + 1 ); o++ ) {
          int plus = tid + o;
          int minus = tid - o;

          clip_plus( clip, length, plus );
          clip_minus( clip, length, minus );

          temp &= data > inp[plus];
          temp &= data >= inp[minus];
        }
        gpu_r[tid] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average 1D kernel (type = %s, order = %d, clip = %d) execution time %f (s)\n", 
           type, order, clip, (time * 1e-9f) / repeat);
  }

  cpu_relextrema_1D<T>(length, order, clip, inp, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++)
    if (cpu_r[i] != gpu_r[i]) {
      error = 1; 
      break;
    }

  free(inp);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("1D test: FAILED\n");
  return time;
}

// length_x is the number of columns
// length_y is the number of rows
template <typename T>
long test_2D (const int length_x, const int length_y, const int order,
              const bool clip, const int axis, const int repeat, const char* type) 
{
  const int length = length_x * length_y;
  T* inp = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    inp[i] = rand() % length;

  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);

  long time;
  #pragma omp target data map(to: inp[0:length]) map(from: gpu_r[0:length])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++)  {
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
      for (int tx = 0; tx < length_y; tx++)
        for (int ty = 0; ty < length_x; ty++) {
          int tid = tx * length_x + ty ;
          const T data = inp[tid] ;
          bool    temp = true ;

          for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus;
            int minus;
            if ( axis == 0 ) {
              plus  = tx + o;
              minus = tx - o;

              clip_plus( clip, length_y, plus );
              clip_minus( clip, length_y, minus );

              plus  = plus * length_x + ty;
              minus = minus * length_x + ty;
            } else {
              plus  = ty + o;
              minus = ty - o;

              clip_plus( clip, length_x, plus );
              clip_minus( clip, length_x, minus );

              plus  = tx * length_x + plus;
              minus = tx * length_x + minus;
            }
            temp &= data > inp[plus] ;
            temp &= data >= inp[minus] ;
          }
          gpu_r[tid] = temp;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average 2D kernel (type = %s, order = %d, clip = %d, axis = %d) execution time %f (s)\n", 
           type, order, clip, axis, (time * 1e-9f) / repeat);
  }

  cpu_relextrema_2D(length_x, length_y, order, clip, axis, inp, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++)
    if (cpu_r[i] != gpu_r[i]) {
      error = 1; 
      break;
    }

  free(inp);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("2D test: FAILED\n");
  return time;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage ./%s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  long time = 0;
  for (int order = 1; order <= 128; order = order * 2) {
    test_1D<   int>(1000000, order, true, repeat, "int");
    test_1D<  long>(1000000, order, true, repeat, "long");
    test_1D< float>(1000000, order, true, repeat, "float");
    test_1D<double>(1000000, order, true, repeat, "double");
  }

  for (int order = 1; order <= 128; order = order * 2) {
    test_2D<   int>(1000, 1000, order, true, 1, repeat, "int");
    test_2D<  long>(1000, 1000, order, true, 1, repeat, "long");
    test_2D< float>(1000, 1000, order, true, 1, repeat, "float");
    test_2D<double>(1000, 1000, order, true, 1, repeat, "double");
  }

  for (int order = 1; order <= 128; order = order * 2) {
    test_2D<   int>(1000, 1000, order, true, 0, repeat, "int");
    test_2D<  long>(1000, 1000, order, true, 0, repeat, "long");
    test_2D< float>(1000, 1000, order, true, 0, repeat, "float");
    test_2D<double>(1000, 1000, order, true, 0, repeat, "double");
  }

  printf("\n-----------------------------------------------\n");
  printf("Total kernel execution time: %lf (s)", time * 1e-9);
  printf("\n-----------------------------------------------\n");

  return 0;
}
