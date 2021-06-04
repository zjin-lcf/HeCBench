#include <cstdio>
#include <cstdlib>

#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold(int *gpuData, int *number, const int len)
{
    int val = 0;

    for (int i = 0; i < len; ++i)
    {
        val += 10;
    }

    if (val != gpuData[0])
    {
        printf("Add failed %d %d\n", val, gpuData[0]);
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val -= 10;
    }

    if (val != gpuData[1])
    {
        printf("Sub failed: %d %d\n", val, gpuData[1]);
    }

    val = -(1<<8);

    for (int i = 0; i < len; ++i)
    {
        val = max(val, number[i]);
    }

    if (val != gpuData[2])
    {
        printf("Max failed: %d %d\n", val, gpuData[2]);
    }

    val = 1 << 8;

    for (int i = 0; i < len; ++i)
    {
        val = min(val, number[i]);
    }

    if (val != gpuData[3])
    {
        printf("Min failed: %d %d\n", val, gpuData[3]);
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        val &= (2 * i + 7);
    }

    if (val != gpuData[4])
    {
        printf("And failed: %d %d\n", val, gpuData[4]);
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val |= (1 << i);
    }

    if (val != gpuData[5])
    {
        printf("Or failed: %d %d\n", val, gpuData[5]);
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        val ^= i;
    }

    if (val != gpuData[6])
    {
        printf("Xor failed %d %d\n", val, gpuData[6]);
    }

    printf("PASS\n");
}

int main()
{
  const int len = 1 << 10;

  srand(2);
  int* number = (int*) malloc(sizeof(int)*len);
  for (int i = 0; i < len; i++) number[i] = rand();

  // add, sub, max, min, and, or, xor
  int gpuData[] = {0, 0, -(1<<8), 1<<8, 0xff, 0, 0xff};

  #pragma omp target data map(tofrom: gpuData[0:7]) map(to:number[0:len])
  {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < len; ++i)
    {
       #pragma omp atomic update  
        gpuData[0] += 10;
       #pragma omp atomic update  
        gpuData[1] -= 10;
       #pragma omp atomic compare  
        gpuData[2] = (gpuData[2] < number[i]) ? number[i] : gpuData[2];
       #pragma omp atomic compare  
        gpuData[3] = (gpuData[3] > number[i]) ? number[i] : gpuData[3];
       #pragma omp atomic update  
        gpuData[4] &= (2*i+7);
       #pragma omp atomic update  
        gpuData[5] |= (1 << i);
       #pragma omp atomic update  
        gpuData[6] ^= i;
    }

    #pragma omp target teams distribute parallel for thread_limit(256) reduction(max: gpuData[2])
    for (int i = 0; i < len; ++i)
       gpuData[2] = max(gpuData[2], number[i]);

    #pragma omp target teams distribute parallel for thread_limit(256) reduction(min: gpuData[3])
    for (int i = 0; i < len; ++i)
       gpuData[3] = min(gpuData[3], number[i]);
  }
  computeGold(gpuData, number, len);
  return 0;
}


