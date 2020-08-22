#include <cstdio>

#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
int
computeGold(int *gpuData, const int len)
{
    int val = 0;

    for (int i = 0; i < len; ++i)
    {
        val += 10;
    }

    if (val != gpuData[0])
    {
        printf("Add failed\n");
        return false;
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val -= 10;
    }

    if (val != gpuData[1])
    {
        printf("Sub failed\n");
        return false;
    }

    val = -(1<<8);

    for (int i = 0; i < len; ++i)
    {
        // fourth element should be len-1
        val = max(val, i);
    }

    if (val != gpuData[2])
    {
        printf("Max failed\n");
        return false;
    }

    val = 1 << 8;

    for (int i = 0; i < len; ++i)
    {
        val = min(val, i);
    }

    if (val != gpuData[3])
    {
        printf("Min failed\n");
        return false;
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        // 9th element should be 1
        val &= (2 * i + 7);
    }

    if (val != gpuData[4])
    {
        printf("And failed\n");
        return false;
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        // 10th element should be 0xff
        val |= (1 << i);
    }

    if (val != gpuData[5])
    {
        printf("Or failed\n");
        return false;
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        // 11th element should be 0xff
        val ^= i;
    }

    if (val != gpuData[6])
    {
        printf("Xor failed\n");
        return false;
    }

    return true;
}

int main()
{
  // add, sub, max, min, and, or, xor
  int gpuData[] = {0, 0, -(1<<8), 1<<8, 0xff, 0, 0xff};

  const int len = 33554432;

#pragma omp target enter data map(to: gpuData[0:7])
{
    #pragma omp target teams distribute parallel for reduction(+:gpuData[0])
    for (int i = 0; i < len; ++i)
    {
        gpuData[0] += 10;
    }

    #pragma omp target teams distribute parallel for reduction(-:gpuData[1])
    for (int i = 0; i < len; ++i)
    {
        gpuData[1] -= 10;
    }

    #pragma omp target teams distribute parallel for reduction(max: gpuData[2])
    for (int i = 0; i < len; ++i)
    {
        gpuData[2] = max(gpuData[2], i);
    }

    #pragma omp target teams distribute parallel for reduction(min: gpuData[3])
    for (int i = 0; i < len; ++i)
    {
        gpuData[3] = min(gpuData[3], i);
    }

    #pragma omp target teams distribute parallel for reduction(&: gpuData[4])
    for (int i = 0; i < len; ++i)
    {
        gpuData[4] = gpuData[4] & (2*i+7);
    }
    #pragma omp target teams distribute parallel for reduction(|: gpuData[5])
    for (int i = 0; i < len; ++i)
    {
        gpuData[5] = gpuData[5] | (1 << i);
    }
    #pragma omp target teams distribute parallel for reduction(^: gpuData[6])
    for (int i = 0; i < len; ++i)
    {
        gpuData[6] = gpuData[6] ^ i;
    }
}
    #pragma omp target exit data map (from: gpuData[0:7])

   computeGold(gpuData, len);
   return 0;
}


