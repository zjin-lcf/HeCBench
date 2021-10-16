#include "hip/hip_runtime.h"
#include <stdio.h>
#include <hip/hip_runtime.h>
#include "kernels.cu"

// check the test result
void check (const unsigned *err_cnt) {
  unsigned err = 0;
  // read error
  hipMemcpy(&err, err_cnt, sizeof(unsigned), hipMemcpyDeviceToHost);

  printf("%s\n", (err != 0) ? "FAIL" : "PASS");

  // reset
  hipMemset(&err, 0, sizeof(unsigned));
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
  dim3 grid (1024);
  dim3 block (64);

  hipLaunchKernelGGL(kernel_write, grid, block, 0, 0, dev_mem, mem_size, p1);

  for(int i = 0; i < 100; i++){
    hipLaunchKernelGGL(kernel_read_write, grid, block, 0, 0, 
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

  hipLaunchKernelGGL(kernel_read, grid, block, 0, 0, dev_mem, mem_size,
      p1, 
      err_cnt,
      err_addr,
      err_expect,
      err_current,
      err_second_read);

  check(err_cnt);
}

int main() {

  unsigned err_count = 0;

  unsigned *err_cnt;
  hipMalloc((void**)&err_cnt, sizeof(unsigned));
  hipMemcpy(err_cnt, &err_count, sizeof(unsigned), hipMemcpyHostToDevice);

  unsigned long *err_addr;
  hipMalloc((void**)&err_addr, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_expect;
  hipMalloc((void**)&err_expect, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_current;
  hipMalloc((void**)&err_current, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_second_read;
  hipMalloc((void**)&err_second_read, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  // 2GB
  unsigned long mem_size = 2*1024*1024*1024UL;
  char *dev_mem;
  hipMalloc((void**)&dev_mem, mem_size);

  printf("test0..\n\n");
  dim3 grid0 (1024);
  dim3 block0 (64);

  hipLaunchKernelGGL(kernel0_write, grid0, block0, 0, 0, dev_mem, mem_size);

  hipLaunchKernelGGL(kernel0_read, grid0, block0, 0, 0, dev_mem, mem_size,
      err_cnt,
      err_addr,
      err_expect,
      err_current,
      err_second_read);

  check(err_cnt);

  printf("test1..\n\n");
  dim3 grid1 (1024);
  dim3 block1 (64);

  hipLaunchKernelGGL(kernel1_write, grid1, block1, 0, 0, dev_mem, mem_size);

  hipLaunchKernelGGL(kernel1_read, grid1, block1, 0, 0, dev_mem, mem_size,
      err_cnt,
      err_addr,
      err_expect,
      err_current,
      err_second_read);

  check(err_cnt);

  printf("test2..\n\n");
  unsigned long p1 = 0;
  unsigned long p2 = ~p1;
  moving_inversion (err_cnt, err_addr, err_expect, err_current,
      err_second_read, dev_mem, mem_size, p1);

  moving_inversion (err_cnt, err_addr, err_expect, err_current,
      err_second_read, dev_mem, mem_size, p2);


  printf("test3..\n\n");
  p1 = 0x8080808080808080;
  p2 = ~p1;
  moving_inversion (err_cnt, err_addr, err_expect, err_current,
      err_second_read, dev_mem, mem_size, p1);

  moving_inversion (err_cnt, err_addr, err_expect, err_current,
      err_second_read, dev_mem, mem_size, p2);

  printf("test4..\n\n");
  srand(123);
  for (int i = 0; i < 20; i++) {
    p1 = rand();
    p1 = (p1 << 32) | rand();
    moving_inversion (err_cnt, err_addr, err_expect, err_current,
        err_second_read, dev_mem, mem_size, p1);
  }

  printf("test5..\n\n");
  dim3 grid5 (64*1024);
  dim3 block5 (64);

  hipLaunchKernelGGL(kernel5_init, grid5, block5, 0, 0, dev_mem, mem_size);

  hipLaunchKernelGGL(kernel5_move, grid5, block5, 0, 0, dev_mem, mem_size);

  hipLaunchKernelGGL(kernel5_check, grid5, block5, 0, 0, dev_mem, mem_size,
      err_cnt,
      err_addr,
      err_expect,
      err_current,
      err_second_read);
  
  check(err_cnt);

  hipFree(err_cnt);
  hipFree(err_addr);
  hipFree(err_expect);
  hipFree(err_current);
  hipFree(err_second_read);
  hipFree(dev_mem);
  return 0;
}
