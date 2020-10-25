#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h> /* defines printf for tests */

typedef unsigned long ulong;

ulong * convertBuffer2Array (char *cbuffer, unsigned size, unsigned step)
{
  unsigned i,j; 
  ulong * values = NULL;
  posix_memalign((void**)&values, 1024, sizeof(ulong)*size/step);
  for (i = 0; i < size / step; i++) {
    values[i] = 0;    // Initialize all elements to zero.
  }
  for (i = 0; i < size; i += step ){
    for (j = 0; j < step; j++){
      values[i / step] += (ulong)((unsigned char)cbuffer[i + j]) << (8*j);
    }
  }
  return values;
}

  
unsigned my_abs ( int x )
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}

unsigned FPCCompress(ulong *values, unsigned size )
{
  unsigned compressable = 0;
  unsigned i;
  for (i = 0; i < size; i++) {
    // 000
    if(values[i] == 0){
      compressable += 1;
      continue;
    }
    // 001 010
    if(my_abs((int)(values[i])) <= 0xFF){
      compressable += 1;
      continue;
    }
    // 011
    if(my_abs((int)(values[i])) <= 0xFFFF){
      compressable += 2;
      continue;
    }
    //100  
    if(((values[i]) & 0xFFFF) == 0 ){
      compressable += 2;
      continue;
    }
    //101
    if( my_abs((int)((values[i]) & 0xFFFF)) <= 0xFF
        && my_abs((int)((values[i] >> 16) & 0xFFFF)) <= 0xFF){
      compressable += 2;
      continue;
    }
    //110
    unsigned byte0 = (values[i]) & 0xFF;
    unsigned byte1 = (values[i] >> 8) & 0xFF;
    unsigned byte2 = (values[i] >> 16) & 0xFF;
    unsigned byte3 = (values[i] >> 24) & 0xFF;
    if(byte0 == byte1 && byte0 == byte2 && byte0 == byte3){
      compressable += 1;
      continue;
    }
    //111
    compressable += 4;
  }
  return compressable;
}


unsigned f1(ulong value, bool* mask) {
  if (value == 0) {
    *mask = 1;
  } 
  return 1;
}

unsigned f2(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFF) *mask = 1;
  return 1;
}

unsigned f3(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFFFF) *mask = 1;
  return 2;
}

unsigned f4(ulong value, bool* mask) {
  if (((value) & 0xFFFF) == 0 ) *mask = 1;
  return 2;
}

unsigned f5(ulong value, bool* mask) {
  if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF && 
      my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) 
    *mask = 1;
  return 2;
}


unsigned f6(ulong value, bool* mask) {
  unsigned byte0 = (value) & 0xFF;
  unsigned byte1 = (value >> 8) & 0xFF;
  unsigned byte2 = (value >> 16) & 0xFF;
  unsigned byte3 = (value >> 24) & 0xFF;
  if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) 
    *mask = 1;
  return 1;
}

unsigned f7(ulong value, bool* mask) {
  *mask = 1;
  return 4;
}

void
fpc_kernel (const ulong* values, unsigned *cmp_size, sycl::nd_item<3> item_ct1,
            unsigned *compressable)
{

  int lid = item_ct1.get_local_id(2);
  int WGS = item_ct1.get_local_range().get(2);
  int gid = item_ct1.get_group(2) * WGS + lid;

  ulong value = values[gid];
  unsigned inc;

  // 000
  if (value == 0){
    inc = 1;
  }
  // 001 010
  else if ((my_abs((int)(value)) <= 0xFF)) {
    inc = 1;
  }
  // 011
  else if ((my_abs((int)(value)) <= 0xFFFF)) {
    inc = 2;
  }
  //100  
  else if ((((value) & 0xFFFF) == 0 )) {
    inc = 2;
  }
  //101
  else if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF
      && my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF ) {
    inc = 2;
  }
  //110
  else if( (((value) & 0xFF) == ((value >> 8) & 0xFF)) &&
      (((value) & 0xFF) == ((value >> 16) & 0xFF)) &&
      (((value) & 0xFF) == ((value >> 24) & 0xFF)) ) {
    inc = 1;
  } else { 
    inc = 4;
  }

    if (lid == 0) *compressable = 0;
  item_ct1.barrier();

  sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
      sycl::local_ptr<unsigned int>(compressable))
      .fetch_add(inc);
  item_ct1.barrier();
  if (lid == WGS-1) {
    sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(cmp_size))
        .fetch_add(*compressable);
  }
}

void
fpc2_kernel (const ulong* values, unsigned *cmp_size, sycl::nd_item<3> item_ct1,
             unsigned *compressable)
{

  int lid = item_ct1.get_local_id(2);
  int WGS = item_ct1.get_local_range().get(2);
  int gid = item_ct1.get_group(2) * WGS + lid;

  unsigned inc;

  bool m1 = 0;
  bool m2 = 0;
  bool m3 = 0;
  bool m4 = 0;
  bool m5 = 0;
  bool m6 = 0;
  bool m7 = 0;

  ulong value = values[gid];
  unsigned inc1 = f1(value, &m1);
  unsigned inc2 = f2(value, &m2);
  unsigned inc3 = f3(value, &m3);
  unsigned inc4 = f4(value, &m4);
  unsigned inc5 = f5(value, &m5);
  unsigned inc6 = f6(value, &m6);
  unsigned inc7 = f7(value, &m7);

  if (m1)
    inc = inc1;
  else if (m2)
    inc = inc2;
  else if (m3)
    inc = inc3;
  else if (m4)
    inc = inc4;
  else if (m5)
    inc = inc5;
  else if (m6)
    inc = inc6;
  else
    inc = inc7;

    if (lid == 0) *compressable = 0;
  item_ct1.barrier();

  sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
      sycl::local_ptr<unsigned int>(compressable))
      .fetch_add(inc);
  item_ct1.barrier();
  if (lid == WGS-1) {
    sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(cmp_size))
        .fetch_add(*compressable);
  }
}

void fpc (const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  *cmp_size_hw = 0;
  ulong* d_values;
  unsigned* d_cmp_size;
  d_values = sycl::malloc_device<ulong>(values_size, q_ct1);
  q_ct1.memcpy(d_values, values, values_size * sizeof(ulong)).wait();
  d_cmp_size = sycl::malloc_device<unsigned int>(1, q_ct1);
  q_ct1.memcpy(d_cmp_size, cmp_size_hw, sizeof(unsigned)).wait();

  sycl::range<3> grids(values_size / wgs, 1, 1);
  sycl::range<3> threads(wgs, 1, 1);

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<unsigned, 0, sycl::access::mode::read_write,
                   sycl::access::target::local>
        compressable_acc_ct1(cgh);

    auto dpct_global_range = grids * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          fpc_kernel(d_values, d_cmp_size, item_ct1,
                     compressable_acc_ct1.get_pointer());
        });
  });

  q_ct1.memcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned)).wait();
  sycl::free(d_values, q_ct1);
  sycl::free(d_cmp_size, q_ct1);
}


void fpc2 (const ulong* values, unsigned *cmp_size_hw, const int values_size, const int wgs)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  *cmp_size_hw = 0;
  ulong* d_values;
  unsigned* d_cmp_size;
  d_values = sycl::malloc_device<ulong>(values_size, q_ct1);
  q_ct1.memcpy(d_values, values, values_size * sizeof(ulong)).wait();
  d_cmp_size = sycl::malloc_device<unsigned int>(1, q_ct1);
  q_ct1.memcpy(d_cmp_size, cmp_size_hw, sizeof(unsigned)).wait();

  sycl::range<3> grids(values_size / wgs, 1, 1);
  sycl::range<3> threads(wgs, 1, 1);

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<unsigned, 0, sycl::access::mode::read_write,
                   sycl::access::target::local>
        compressable_acc_ct1(cgh);

    auto dpct_global_range = grids * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          fpc2_kernel(d_values, d_cmp_size, item_ct1,
                      compressable_acc_ct1.get_pointer());
        });
  });

  q_ct1.memcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned)).wait();
  sycl::free(d_values, q_ct1);
  sycl::free(d_cmp_size, q_ct1);
}


int main(int argc, char** argv) {

  // size must be a multiple of step and work-group size (wgs)
  const int step = 4;
  const int size = atoi(argv[1]);
  const int wgs = atoi(argv[2]);

  // create the char buffer
  char* cbuffer = (char*) malloc (sizeof(char) * size * step);

  srand(2);
  for (int i = 0; i < size*step; i++) {
    cbuffer[i] = 0xFF << (rand() % 256);
  }

  ulong *values = convertBuffer2Array (cbuffer, size, step);
  unsigned values_size = size / step;

  // run on the host
  unsigned cmp_size = FPCCompress(values, values_size);

  unsigned cmp_size_hw; 

  // fpc is faster than fpc2 on a GPU
  for (int i = 0; i < 100; i++) {
    fpc(values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size);
    }
    fpc2(values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size);
    }
  }

  free(values);
  free(cbuffer);
  return 0;
}
