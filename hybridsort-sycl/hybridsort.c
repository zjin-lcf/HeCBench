
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "bucketsort.h"
#include "mergesort.h"
#include <time.h>
#define TIMER 

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define SIZE (50000000)

////////////////////////////////////////////////////////////////////////////////
int compare(const void *a, const void *b) {
  if(*((float *)a) < *((float *)b)) return -1;
  else if(*((float *)a) > *((float *)b)) return 1;
  else return 0;
}

////////////////////////////////////////////////////////////////////////////////
sycl::float4* runMergeSort(int listsize, int divisions,
    sycl::float4 *d_origList, sycl::float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets);

int main(int argc, char** argv)
{
  // Fill our data set with random float values
  int numElements = 0 ;

  if(strcmp(argv[1],"r") ==0) {
    numElements = SIZE;
  }
  else {
    FILE *fp;
    fp = fopen(argv[1],"r");
    if(fp == NULL) {
      printf("Error reading file \n");
      exit(EXIT_FAILURE);
    }
    int count = 0;
    float c;

    while(fscanf(fp,"%f",&c) != EOF) {
      count++;
    }
    fclose(fp);

    numElements = count;
  }
  printf("Sorting list of %d floats.\n", numElements);
  int mem_size = (numElements + (DIVISIONS*4))*sizeof(float);
  // Allocate enough for the input list
  float *cpu_idata = (float *)malloc(mem_size);
  float *cpu_odata = (float *)malloc(mem_size);
  // Allocate enough for the output list on the cpu side
  float *d_output = (float *)malloc(mem_size);
  // Allocate enough memory for the output list on the gpu side
  float *gpu_odata; // = (float *)malloc(mem_size);
  float datamin = FLT_MAX;
  float datamax = -FLT_MAX;

  if(strcmp(argv[1],"r")==0) {
    for (int i = 0; i < numElements; i++) {
      // Generate random floats between 0 and 1 for the input data
      //cpu_idata[i] = (float) rand() / RAND_MAX;
      cpu_idata[i] = SIZE-i; //(float) rand() / RAND_MAX;

      //Compare data at index to data minimum, if less than current minimum, set that element as new minimum
      datamin = fminf(cpu_idata[i], datamin);
      //Same as above but for maximum
      datamax = fmaxf(cpu_idata[i], datamax);
    }
  }
  else {
    FILE *fp;
    fp = fopen(argv[1],"r");
    for(int i = 0; i < numElements; i++) {
      fscanf(fp,"%f",&cpu_idata[i]);
      datamin = fminf(cpu_idata[i], datamin);
      datamax = fmaxf(cpu_idata[i],datamax);
    }
  }
  FILE *tp;
  const char filename2[]="./hybridinput.txt";
  tp = fopen(filename2,"w");
  for(int i = 0; i < SIZE; i++) {
    fprintf(tp,"%f ",cpu_idata[i]); 
  }

  fclose(tp);
  memcpy(cpu_odata, cpu_idata, mem_size);

  int *sizes = (int*) malloc(DIVISIONS * sizeof(int));
  int *nullElements = (int*) malloc(DIVISIONS * sizeof(int));
  unsigned int *origOffsets = (unsigned int *) malloc((DIVISIONS + 1) * sizeof(int));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v);
#else
  sycl::queue q(sycl::cpu_selector_v);
#endif

  // time bucketsort
  clock_t bucketsort_start = clock();
  bucketSort(q, cpu_idata,d_output,numElements,sizes,nullElements,datamin,datamax, origOffsets);
  clock_t bucketsort_diff = clock() - bucketsort_start;
 
  sycl::float4 *d_origList = (sycl::float4*) d_output;
  sycl::float4 *d_resultList = (sycl::float4*) cpu_idata;

  int newlistsize = 0;
  for(int i = 0; i < DIVISIONS; i++){
    newlistsize += sizes[i] * 4;
  }

  // time mergesort
  clock_t mergesort_start = clock();
  sycl::float4 *mergeresult = runMergeSort(q, newlistsize,DIVISIONS,d_origList,d_resultList,sizes,nullElements,origOffsets);
  clock_t mergesort_diff = clock() - mergesort_start;
  gpu_odata = (float*)mergeresult;

#ifdef TIMER
  float bucketsort_msec = bucketsort_diff * 1000 / CLOCKS_PER_SEC;
  float mergesort_msec = mergesort_diff * 1000 / CLOCKS_PER_SEC;


  printf("GPU execution time: %0.3f ms  \n", bucketsort_msec+mergesort_msec);
  printf("  --Bucketsort execution time: %0.3f ms \n", bucketsort_msec);
  printf("  --Mergesort execution time: %0.3f ms \n", mergesort_msec);
#endif


  // always verifiy
  clock_t cpu_start = clock(), cpu_diff;

  qsort(cpu_odata, numElements, sizeof(float), compare);
  cpu_diff = clock() - cpu_start;
  float cpu_msec = cpu_diff * 1000.0f / CLOCKS_PER_SEC;
  printf("CPU execution time: %0.3f ms  \n", cpu_msec);
  printf("Checking result...");

  // Result checking
  int count = 0;
  for(int i = 0; i < numElements; i++){
    if(cpu_odata[i] != gpu_odata[i])
    {
      printf("Sort missmatch on element %d: \n", i);
      printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i]);
      count++;
      break;
    }
  }
  if(count == 0) printf("PASSED.\n");
  else printf("FAILED.\n");

#ifdef OUTPUT
  FILE *tp1;
  const char filename3[]="./hybridoutput.txt";
  tp1 = fopen(filename3,"w");
  for(int i = 0; i < SIZE; i++) {
    fprintf(tp1,"%f ",cpu_idata[i]);
  }

  fclose(tp1);
#endif

  free(cpu_idata);
  free(cpu_odata);
  free(d_output);
  free(sizes);
  free(nullElements);
  free(origOffsets);

  return 0;
}
