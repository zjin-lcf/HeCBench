template<typename divisor_type>
void throughput_test(
    const int n,
    divisor_type d1,
    divisor_type d2,
    divisor_type d3,
    int dummy,
    int * buf)
{
  #pragma omp target teams distribute parallel for thread_limit(256) \
   map(to: n, d1, d2, d3, dummy) map(alloc: buf[0:1]) 
  for (int x = 0; x < n; x++) {
    int x1 = x / d1;
    int x2 = x / d2;
    int x3 = x / d3;
    int aggregate = x1 + x2 + x3;  
    if (aggregate & dummy == 1) buf[0] = aggregate;
  }
}

template<typename divisor_type>
void latency_test(
    const int n,
    divisor_type d1,
    divisor_type d2,
    divisor_type d3,
    divisor_type d4,
    divisor_type d5,
    divisor_type d6,
    divisor_type d7,
    divisor_type d8,
    divisor_type d9,
    divisor_type d10,
    int dummy,
    int * buf)
{
  #pragma omp target teams distribute parallel for thread_limit(256) \
   map(to: n, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, dummy) map(alloc: buf[0:1]) 
  for (int x = 0; x < n; x++) {
    x /= d1;
    x /= d2;
    x /= d3;
    x /= d4;
    x /= d5;
    x /= d6;
    x /= d7;
    x /= d8;
    x /= d9;
    x /= d10;
    if (x & dummy == 1) buf[0] = x;
  }
}

void check(const int n, int_fastdiv divisor, int * results)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int divident = 0; divident < n; divident++) { 

    int quotient = divident / (int)divisor;
    int fast_quotient = divident / divisor;

    if (quotient != fast_quotient)
    {
      int error_id;
      #pragma omp atomic capture
      error_id = results[0]++;
      if (error_id == 0)
      {
        results[1] = divident;
        results[2] = quotient;
        results[3] = fast_quotient;
      }
    }

    divident = -divident;
    quotient = divident / (int)divisor;
    fast_quotient = divident / divisor;

    if (quotient != fast_quotient)
    {
      int error_id;
      #pragma omp atomic capture
      error_id = results[0]++;
      if (error_id == 0)
      {
        results[1] = divident;
        results[2] = quotient;
        results[3] = fast_quotient;
      }
    }
  }
}
