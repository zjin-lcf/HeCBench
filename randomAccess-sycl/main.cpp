#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "common.h"

typedef unsigned long long int u64Int;
typedef long long int s64Int;

/* Random number generator */
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

#define NUPDATE (4 * TableSize)

#define K1_BLOCKSIZE  256
#define K2_BLOCKSIZE  128
#define K3_BLOCKSIZE  128

u64Int
HPCC_starts(s64Int n)
{
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;

  #pragma unroll
  for (i=0; i<64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
  }

  for (i=62; i>=0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    #pragma unroll
    for (j=0; j<64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
  }

  return ran;
}


int main(int argc, char** argv) {
  //double GUPs;
  int failure;
  u64Int i;
  u64Int temp;
  //double cputime;               /* CPU time to update table */
  //double realtime;              /* Real time to update table */
  double totalMem;
  u64Int *Table = NULL;
  u64Int logTableSize, TableSize;

  /* calculate local memory per node for the update table */
  totalMem = 1024*1024*512;
  totalMem /= sizeof(u64Int);

  /* calculate the size of update array (must be a power of 2) */
  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ; /* EMPTY */

   printf("Table size = %llu\n",  TableSize);

   posix_memalign((void**)&Table, 1024, TableSize * sizeof(u64Int));

  if (! Table ) {
    fprintf( stderr, "Failed to allocate memory for the update table %llu\n", TableSize);
    return 1;
  }

  /* Print parameters for run */
  fprintf( stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize,TableSize);
  fprintf( stdout, "Number of updates = %llu\n", NUPDATE);

#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<u64Int, 1> d_Table (TableSize);
  buffer<u64Int, 1> d_ran (128);
  range<1> global_work_size ((TableSize+K1_BLOCKSIZE-1) / K1_BLOCKSIZE * K1_BLOCKSIZE);

  q.submit([&](handler &h) {
    auto Table = d_Table.get_access<sycl_write>(h);
    h.parallel_for<class init_table>(nd_range<1>(range<1>(global_work_size), range<1>(K1_BLOCKSIZE)), [=](nd_item<1> item) {
      int i = item.get_global_id(0);
      if (i < TableSize) Table[i] = i;
    });
  });

  q.submit([&](handler &h) {
    auto ran = d_ran.get_access<sycl_write>(h);
    h.parallel_for<class init_ranarray>(nd_range<1>(range<1>(K2_BLOCKSIZE), range<1>(K2_BLOCKSIZE)), [=](nd_item<1> item) {
      int j = item.get_global_id(0);
      ran[j] = HPCC_starts ((NUPDATE/128) * j);
    });
  });

  q.submit([&](handler &h) {
    auto Table = d_Table.get_access<sycl_atomic>(h);
    auto ran = d_ran.get_access<sycl_read_write>(h);
    h.parallel_for<class update>(nd_range<1>(range<1>(K3_BLOCKSIZE), range<1>(K3_BLOCKSIZE)), [=](nd_item<1> item) {
      int j = item.get_global_id(0);
      for (u64Int i=0; i<NUPDATE/128; i++) {
        ran[j] = (ran[j] << 1) ^ ((s64Int) ran[j] < 0 ? POLY : 0);
        atomic_fetch_xor(Table[ran[j] & (TableSize-1)], ran[j]);
      }
    });
  });

  q.submit([&](handler &h) {
    auto Table_acc = d_Table.get_access<sycl_read>(h);
    h.copy(Table_acc, Table);
  });
  q.wait();

  /* validation */
  temp = 0x1;
  for (i=0; i<NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int) temp < 0) ? POLY : 0);
    Table[temp & (TableSize-1)] ^= temp;
  }
  
  temp = 0;
  for (i=0; i<TableSize; i++)
    if (Table[i] != i) {
      temp++;
    }

  fprintf( stdout, "Found %llu errors in %llu locations (%s).\n",
           temp, TableSize, (temp <= 0.01*TableSize) ? "passed" : "failed");
  if (temp <= 0.01*TableSize) failure = 0;
  else failure = 1;

  free( Table );
  return failure;

}

