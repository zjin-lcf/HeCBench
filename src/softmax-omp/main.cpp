#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256

// A C model derived from the OpenCL kernel 
void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest) {
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 1; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

// begin of softMax
void softMax (const int numTeams, const int numThreads,
              const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
      max_ = fmaxf(max_, src[i * sliceSize + j]);
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      sum += expf(src[i * sliceSize + j] - max_);
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
    }
  }
}
// end of softMax

void softMax2 (const int numTeams, const int numThreads,
               const int numSlice, const int sliceSize,
               const float* src, float* dest)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    #pragma omp parallel for reduction(max:max_) num_threads(numThreads)
    for (int j = 1; j < sliceSize; j++) {
      max_ = fmaxf(max_, src[i * sliceSize + j]);
    }
    float sum = 0;
    #pragma omp parallel for reduction(+:sum) num_threads(numThreads)
    for (int j = 0; j < sliceSize; j++) {
      sum += expf(src[i * sliceSize + j] - max_);
    }
    #pragma omp parallel for num_threads(numThreads)
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of slices> <slice size> <implementations> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: optimized\n");
    return 1;
  }
   
  int numSlice = atoi(argv[1]);
  int sliceSize = atoi(argv[2]);
  int kernel = atoi(argv[3]);
  int repeat = atoi(argv[4]);
  int numElem = numSlice * sliceSize;

  float* input = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_gpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_cpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);

  srand(2);
  for (int i = 0; i < numSlice; i++)
    for (int j = 0; j < sliceSize; j++)
      input[i*sliceSize+j] = rand() % 13; 

  #pragma omp target data map(to: input[0:numElem]) map(from: output_gpu[0:numElem])
  {
    if (kernel == 1) {
      const int numTeams = (numSlice+BLOCK_SIZE/32-1)/(BLOCK_SIZE/32);
      const int numThreads = 32;

      auto start = std::chrono::steady_clock::now();
    
      for (int n = 0; n < repeat; n++) {
        softMax2(numTeams, numThreads, numSlice, sliceSize, input, output_gpu);
      }
    
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
    }
    else {
      const int numTeams = (numSlice+BLOCK_SIZE-1)/BLOCK_SIZE;
      const int numThreads = BLOCK_SIZE;

      auto start = std::chrono::steady_clock::now();
    
      for (int n = 0; n < repeat; n++) {
        softMax(numTeams, numThreads, numSlice, sliceSize, input, output_gpu);
      }
    
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
    }
  }

  // verification
  bool ok = true;
  softMax_cpu(numSlice, sliceSize, input, output_cpu);
  for (int i = 0; i < numElem; i++) {
    if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-3) {
      printf("@index %d host: %f device: %f\n", i, output_cpu[i], output_gpu[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(input);
  free(output_cpu);
  free(output_gpu);
  return 0;
}
