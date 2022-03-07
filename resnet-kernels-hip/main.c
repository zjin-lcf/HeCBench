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

  int i;
  int sum = 0;
  for (i = 0; i < repeat; i++) {
    int res = -1;
    switch (mode) {
      case 0:
        res = kernel_128();
        break;
      case 1:
        res = kernel_256();
        break;
      case 2:
        res = kernel_128_1_in();
        break;
      case 3:
        res = kernel_128_1_out();
        break;
      case 4:
        res = kernel_256_1_in();
        break;
      case 5:
        res = kernel_256_1_out();
        break;
    }
    if (i > 1) sum += res >> 16;
  }
  printf("Case %d: Average Time: [%d us]\n", mode, sum / (repeat - 2));
  
  return 0;
}
