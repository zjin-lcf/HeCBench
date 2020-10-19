#include <stdio.h>

typedef int inta[10];

void f (int* a, int s) {
	int i;
	for (i=0;i<s;i++)
		printf("%d ", a[i]);
}

int main() {
  inta a;
  a[0]=1;
  a[1]=2;
  a[9]=8;
  f((int*)&a, 10);
  f(a, 10);
  // cannot convert ‘int (*)[10]’ to ‘int*’ for argument ‘1’ to ‘void f(int*, int)’
  //f(&a, 10);
  return 0;
}
