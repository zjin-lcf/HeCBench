#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "kernels.h"

// check the test result
void check (unsigned *err_cnt) {
  // read error
  #pragma omp target update from (err_cnt[0:1])

  printf("%s", err_cnt[0] ? "x" : ".");

  // reset
  #pragma omp target 
  err_cnt[0] = 0;
}

// moving inversion tests with complementary patterns
void moving_inversion (
    unsigned *err_cnt,
    unsigned long *err_addr,
    unsigned long *err_expect,
    unsigned long *err_current,
    unsigned long *err_second_read,
    char *dev_mem,
    unsigned long mem_size,
    unsigned long p1)
{
  unsigned long p2 = ~p1;

  kernel_write(dev_mem, mem_size, p1);

  for(int i = 0; i < 10; i++){
    kernel_read_write(
        dev_mem, 
        mem_size,
        p1, p2,
        err_cnt,
        err_addr,
        err_expect,
        err_current,
        err_second_read);
    p1 = p2;
    p2 = ~p1;
  }

  kernel_read(dev_mem, mem_size,
      p1, 
      err_cnt,
      err_addr,
      err_expect,
      err_current,
      err_second_read);

  check(err_cnt);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned err_cnt[1] = {0};
  unsigned long *err_addr = (unsigned long*) malloc (sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
  unsigned long *err_expect = (unsigned long*) malloc (sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
  unsigned long *err_current = (unsigned long*) malloc (sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
  unsigned long *err_second_read = (unsigned long*) malloc (sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);

  // 2GB
  unsigned long mem_size = 2*1024*1024*1024UL;
  char *dev_mem = (char*) malloc (mem_size);

  #pragma omp target data map(to:err_cnt[0:1]) \
                          map(alloc: err_addr[0:MAX_ERR_RECORD_COUNT], \
                                     err_expect[0:MAX_ERR_RECORD_COUNT], \
                                     err_current[0:MAX_ERR_RECORD_COUNT], \
                                     err_second_read[0:MAX_ERR_RECORD_COUNT], \
                                     dev_mem[0:mem_size])
  {
    printf("\ntest0: ");

    for (int i = 0; i < repeat; i++) {
      kernel0_write(dev_mem, mem_size);

      kernel0_read(dev_mem, mem_size,
          err_cnt,
          err_addr,
          err_expect,
          err_current,
          err_second_read);
    }

    check(err_cnt);

    printf("\ntest1: ");

    for (int i = 0; i < repeat; i++) {
      kernel1_write(dev_mem, mem_size);

      kernel1_read(dev_mem, mem_size,
          err_cnt,
          err_addr,
          err_expect,
          err_current,
          err_second_read);
    }

    check(err_cnt);

    printf("\ntest2: ");
    for (int i = 0; i < repeat; i++) {
      unsigned long p1 = 0;
      unsigned long p2 = ~p1;
      moving_inversion (err_cnt, err_addr, err_expect, err_current,
          err_second_read, dev_mem, mem_size, p1);

      moving_inversion (err_cnt, err_addr, err_expect, err_current,
          err_second_read, dev_mem, mem_size, p2);
    }

    printf("\ntest3: ");
    for (int i = 0; i < repeat; i++) {
      unsigned long p1 = 0x8080808080808080;
      unsigned long p2 = ~p1;
      moving_inversion (err_cnt, err_addr, err_expect, err_current,
          err_second_read, dev_mem, mem_size, p1);
  
      moving_inversion (err_cnt, err_addr, err_expect, err_current,
          err_second_read, dev_mem, mem_size, p2);
    }
  
    printf("\ntest4: ");
    srand(123);
    for (int i = 0; i < repeat; i++) {
      unsigned long p1 = rand();
      p1 = (p1 << 32) | rand();
      moving_inversion (err_cnt, err_addr, err_expect, err_current,
          err_second_read, dev_mem, mem_size, p1);
    }

    printf("\ntest5: ");

    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < repeat; i++) {
      kernel5_init(dev_mem, mem_size);

      kernel5_move(dev_mem, mem_size);

      kernel5_check(dev_mem, mem_size,
          err_cnt,
          err_addr,
          err_expect,
          err_current,
          err_second_read);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    check(err_cnt);

    printf("\nAverage kernel execution time (test5): %f (s)\n", (time * 1e-9f) / repeat);
  }

  free(err_addr);
  free(err_expect);
  free(err_current);
  free(err_second_read);
  free(dev_mem);
  return 0;
}
