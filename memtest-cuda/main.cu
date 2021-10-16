#include <stdio.h>
#include <cuda.h>
#include "kernels.cu"

// check the test result
void check (unsigned *err_cnt) {
  unsigned err = 0;
  // read error
  cudaMemcpy(&err, err_cnt, sizeof(unsigned), cudaMemcpyDeviceToHost);

  printf("%s\n", (err != 0) ? "FAIL" : "PASS");

  // reset
  cudaMemset(&err, 0, sizeof(unsigned));
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

  kernel_write(dev_mem, mem_size, p1);

  for(int i = 0; i < 100; i++){
    kernel_read_write <<<grid, block>> (
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

  kernel_read <<<grid, block>>> (mem, mem_size,
      p1, 
      err_cnt,
      err_addr,
      err_expect,
      err_current,
      err_read);
  check(q, err_cnt);
}

int main() {

  unsigned err_count = 0;

  unsigned *err_cnt;
  cudaMalloc((void**)&err_cnt, sizeof(unsigned));
  cudaMemcpy(err_cnt, &err_count, sizeof(unsigned), cudaMemcpyHostToDevice);

  unsigned long *err_addr;
  cudaMalloc((void**)&err_addr, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_expect;
  cudaMalloc((void**)&err_expect, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_current;
  cudaMalloc((void**)&err_current, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_second_read;
  cudaMalloc((void**)&err_second_read, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  // 2GB
  unsigned long mem_size = 2*1024*1024*1024UL;
  char *dev_mem;
  cudaMalloc((void**)&dev_mem, mem_size);

  printf("test0..\n\n");
  dim3 grid0 (1024);
  dim3 block0 (64);

  kernel0_write <<<grid0, block0>>> (dev_mem, mem_size);

  kernel0_read <<<grid0, block0>>> (dev_mem, mem_size,
      err_cnt,
      err_address,
      err_expect,
      err_current,
      err_second_read);

  check(q, err_cnt);

  printf("test1..\n\n");
  dim3 grid1 (1024);
  dim3 block1 (64);

  kernel1_write <<<grid1, block1>>> (dev_mem, mem_size);

  kernel1_read <<<grid1, block1>>> (dev_mem, mem_size,
      err_cnt,
      err_address,
      err_expect,
      err_current,
      err_second_read);
  check(q, err_cnt);

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

  kernel5_init <<<grid5, block5>>> (dev_mem, mem_size);

  kernel5_move <<<grid5, block5>>> (dev_mem, mem_size);

  kernel5_check <<<grid5, block5>>> (dev_mem, mem_size,
      err_cnt,
      err_address,
      err_expect,
      err_current,
      err_second_read);
  check(q, err_cnt);

  cudaFree(err_cnt);
  cudaFree(err_address);
  cudaFree(err_expect);
  cudaFree(err_current);
  cudaFree(err_second_read);
  cudaFree(dev_mem);
  return 0;
}
