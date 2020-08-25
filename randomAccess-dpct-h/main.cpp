#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef unsigned long long int u64Int;
typedef long long int s64Int;

/* Random number generator */
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

#define NUPDATE (4 * TableSize)

/* CUDA specific parameters */
#define K1_BLOCKSIZE  256
#define K2_BLOCKSIZE  128
#define K3_BLOCKSIZE  128


u64Int HPCC_starts(s64Int n)
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

void initTable (u64Int* Table, const u64Int TableSize,
                sycl::nd_item<3> item_ct1) {
  int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
  if (i < TableSize) Table[i] = i;
}

void initRan (u64Int* ran, const u64Int TableSize, sycl::nd_item<3> item_ct1) {
  int j = item_ct1.get_local_id(2);
  ran[j] = HPCC_starts ((NUPDATE/128) * j);
}

void update (u64Int* Table, u64Int* ran, const u64Int TableSize,
             sycl::nd_item<3> item_ct1) {
  int j = item_ct1.get_local_id(2);
  for (u64Int i=0; i<NUPDATE/128; i++) {
    ran[j] = (ran[j] << 1) ^ ((s64Int) ran[j] < 0 ? POLY : 0);
    sycl::atomic<u64Int>(
        sycl::global_ptr<u64Int>(&Table[ran[j] & (TableSize - 1)]))
        .fetch_xor(ran[j]);
  }
}

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
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

  u64Int* d_Table;
  dpct::dpct_malloc((void **)&d_Table, TableSize * sizeof(u64Int));

  u64Int *d_ran;
  dpct::dpct_malloc((void **)&d_ran, 128 * sizeof(u64Int));

  /* initialize the table */
  {
    dpct::buffer_t d_Table_buf_ct0 = dpct::get_buffer(d_Table);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_Table_acc_ct0 =
          d_Table_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1,
                             (TableSize + K1_BLOCKSIZE - 1) / K1_BLOCKSIZE) *
                  sycl::range<3>(1, 1, K1_BLOCKSIZE),
              sycl::range<3>(1, 1, K1_BLOCKSIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            initTable((u64Int *)(&d_Table_acc_ct0[0]), TableSize, item_ct1);
          });
    });
  }

  /* initialize the ran structure */
  {
    dpct::buffer_t d_ran_buf_ct0 = dpct::get_buffer(d_ran);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_ran_acc_ct0 =
          d_ran_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, K2_BLOCKSIZE),
                                         sycl::range<3>(1, 1, K2_BLOCKSIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         initRan((u64Int *)(&d_ran_acc_ct0[0]), TableSize,
                                 item_ct1);
                       });
    });
  }

  /* update the table */
  {
    dpct::buffer_t d_Table_buf_ct0 = dpct::get_buffer(d_Table);
    dpct::buffer_t d_ran_buf_ct1 = dpct::get_buffer(d_ran);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_Table_acc_ct0 =
          d_Table_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto d_ran_acc_ct1 =
          d_ran_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, K3_BLOCKSIZE),
                                         sycl::range<3>(1, 1, K3_BLOCKSIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                         update((u64Int *)(&d_Table_acc_ct0[0]),
                                (u64Int *)(&d_ran_acc_ct1[0]), TableSize,
                                item_ct1);
                       });
    });
  }

  dpct::dpct_memcpy(Table, d_Table, TableSize * sizeof(u64Int),
                    dpct::device_to_host);

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
  dpct::dpct_free(d_Table);
  dpct::dpct_free(d_ran);
  return failure;

}


