#include "util.h"
#include "Kernel128_one.h"
#include "Kernel128_winograd.h"
#include "Kernel256_one.h"
#include "Kernel256_winograd.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage %s <mode> <repeat more than twice>\n", argv[0]);
    return 1;
  }

  const int mode = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  int i;
  int sum = 0;
  for (i = 0; i < repeat; i++) {
    int res = -1;
    switch (mode) {
      case 0:
        res = kernel_128(q);
        break;
      case 1:
        res = kernel_256(q);
        break;
      case 2:
        res = kernel_128_1_in(q);
        break;
      case 3:
        res = kernel_128_1_out(q);
        break;
      case 4:
        res = kernel_256_1_in(q);
        break;
      case 5:
        res = kernel_256_1_out(q);
        break;
    }
    if (i > 1) sum += res >> 16;
  }
  printf("Case %d: Average Time: [%d us]\n", mode, sum / (repeat - 2));
  return 0;
}
