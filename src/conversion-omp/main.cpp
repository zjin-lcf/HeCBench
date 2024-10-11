#include <algorithm>
#include <chrono>
#include <cstdio>
#include <omp.h>

typedef unsigned char uchar;

template <typename Td, typename Ts>
void convert(int nelems, int niters)
{
  Ts *src = (Ts*) malloc (nelems * sizeof(Ts));
  Td *dst = (Td*) malloc (nelems * sizeof(Td));

  const int ls = std::min(nelems, 256);
  const int gs = (nelems + ls - 1) / ls;

  #pragma omp target data map(alloc: src[0:nelems], dst[0:nelems])
  {
    // Warm-up run
    #pragma omp target teams distribute parallel for num_teams(gs) num_threads(ls) 
    for (int i = 0; i < nelems; i++) {
      dst[i] = static_cast<Td>(src[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < niters; i++) {
      #pragma omp target teams distribute parallel for num_teams(gs) num_threads(ls) 
      for (int i = 0; i < nelems; i++)
        dst[i] = static_cast<Td>(src[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>
                  (end - start).count() / niters / 1.0e6;
    double size = (sizeof(Td) + sizeof(Ts)) * nelems / 1e9;
    printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size, time, size / time);
  }
  free(src);
  free(dst);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int niters = atoi(argv[2]);

/*
  printf("bfloat16 -> half\n");
  convert<half, __nv_bfloat16>(nelems, niters); 
  printf("bfloat16 -> float\n");
  convert<float, __nv_bfloat16>(nelems, niters); 
  printf("bfloat16 -> int\n");
  convert<int, __nv_bfloat16>(nelems, niters); 
  printf("bfloat16 -> char\n");
  convert<char, __nv_bfloat16>(nelems, niters); 
  printf("bfloat16 -> uchar\n");
  convert<uchar, __nv_bfloat16>(nelems, niters); 

  printf("half -> half\n");
  convert<half, half>(nelems, niters); 
  printf("half -> float\n");
  convert<float, half>(nelems, niters); 
  printf("half -> int\n");
  convert<int, half>(nelems, niters); 
  printf("half -> char\n");
  convert<char, half>(nelems, niters); 
  printf("half -> uchar\n");
  convert<uchar, half>(nelems, niters); 
*/

  printf("float -> float\n");
  convert<float, float>(nelems, niters); 
  //printf("float -> half\n");
  //convert<half, float>(nelems, niters); 
  printf("float -> int\n");
  convert<int, float>(nelems, niters); 
  printf("float -> char\n");
  convert<char, float>(nelems, niters); 
  printf("float -> uchar\n");
  convert<uchar, float>(nelems, niters); 

  printf("int -> int\n");
  convert<int, int>(nelems, niters); 
  printf("int -> float\n");
  convert<float, int>(nelems, niters); 
  //printf("int -> half\n");
  //convert<half, int>(nelems, niters); 
  printf("int -> char\n");
  convert<char, int>(nelems, niters); 
  printf("int -> uchar\n");
  convert<uchar, int>(nelems, niters); 

  printf("char -> int\n");
  convert<int, char>(nelems, niters); 
  printf("char -> float\n");
  convert<float, char>(nelems, niters); 
  //printf("char -> half\n");
  //convert<half, char>(nelems, niters); 
  printf("char -> char\n");
  convert<char, char>(nelems, niters); 
  printf("char -> uchar\n");
  convert<uchar, char>(nelems, niters); 

  printf("uchar -> int\n");
  convert<int, uchar>(nelems, niters); 
  printf("uchar -> float\n");
  convert<float, uchar>(nelems, niters); 
  //printf("uchar -> half\n");
  //convert<half, uchar>(nelems, niters); 
  printf("uchar -> char\n");
  convert<char, uchar>(nelems, niters); 
  printf("uchar -> uchar\n");
  convert<uchar, uchar>(nelems, niters); 

  return 0;
}
