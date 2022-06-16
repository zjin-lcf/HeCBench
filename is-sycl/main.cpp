#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"
#include "is.h"
#include "kernels.h"

/**********************/
/* partial verif info */
/**********************/
int test_index_array[TEST_ARRAY_SIZE],
    test_rank_array[TEST_ARRAY_SIZE],

    S_test_index_array[TEST_ARRAY_SIZE] = {48427,17148,23627,62548,4431},
    S_test_rank_array[TEST_ARRAY_SIZE] = {0,18,346,64917,65463},

    W_test_index_array[TEST_ARRAY_SIZE] = {357773,934767,875723,898999,404505},
    W_test_rank_array[TEST_ARRAY_SIZE] = {1249,11698,1039987,1043896,1048018},

    A_test_index_array[TEST_ARRAY_SIZE] = {2112377,662041,5336171,3642833,4250760},
    A_test_rank_array[TEST_ARRAY_SIZE] = {104,17523,123928,8288932,8388264},

    B_test_index_array[TEST_ARRAY_SIZE] = {41869,812306,5102857,18232239,26860214},
    B_test_rank_array[TEST_ARRAY_SIZE] = {33422937,10244,59149,33135281,99}, 

    C_test_index_array[TEST_ARRAY_SIZE] = {44172927,72999161,74326391,129606274,21736814},
    C_test_rank_array[TEST_ARRAY_SIZE] = {61147,882988,266290,133997595,133525895},

    D_test_index_array[TEST_ARRAY_SIZE] = {1317351170,995930646,1157283250,1503301535,1453734525},
    D_test_rank_array[TEST_ARRAY_SIZE] = {1,36538729,1978098519,2145192618,2147425337};

/* is */
int main(int argc, char** argv){
  /* printout initial NPB info */
  printf("\n\n NAS Parallel Benchmarks 4.1 IS Benchmark\n\n");
  printf(" Size:  %ld  (class %c)\n", (long)TOTAL_KEYS, CLASS);
  printf(" Iterations:   %d\n", MAX_ITERATIONS);

  if (argc != 4) {
    printf("Usage: %s <threads per block for the create_seq kernel>\n", argv[0]);
    printf("           <threads per block for the rank kernel>\n");
    printf("           <threads per block for the verify kernel>\n");
    return 1;
  }

  int i, iteration;
  int passed_verification;
  int size_shared_data_on_rank_4;
  int size_shared_data_on_rank_5;
  int size_shared_data_on_full_verify_3;
  int threads_per_block_on_create_seq;
  int threads_per_block_on_rank;
  int threads_per_block_on_rank_1;
  int threads_per_block_on_rank_2;
  int threads_per_block_on_rank_3;
  int threads_per_block_on_rank_4;
  int threads_per_block_on_rank_5;
  int threads_per_block_on_rank_6;
  int threads_per_block_on_rank_7;
  int threads_per_block_on_full_verify;
  int threads_per_block_on_full_verify_1;
  int threads_per_block_on_full_verify_2;
  int threads_per_block_on_full_verify_3;
  int blocks_per_grid_on_create_seq;
  int blocks_per_grid_on_rank_1;
  int blocks_per_grid_on_rank_2;
  int blocks_per_grid_on_rank_3;
  int blocks_per_grid_on_rank_4;
  int blocks_per_grid_on_rank_5;
  int blocks_per_grid_on_rank_6;
  int blocks_per_grid_on_rank_7;
  int blocks_per_grid_on_full_verify_1;
  int blocks_per_grid_on_full_verify_2;
  int blocks_per_grid_on_full_verify_3;
  int amount_of_work_on_create_seq;
  int amount_of_work_on_rank_1;
  int amount_of_work_on_rank_2;
  int amount_of_work_on_rank_3;
  int amount_of_work_on_rank_4;
  int amount_of_work_on_rank_5;
  int amount_of_work_on_rank_6;
  int amount_of_work_on_rank_7;
  int amount_of_work_on_full_verify_1;
  int amount_of_work_on_full_verify_2;
  int amount_of_work_on_full_verify_3;

  /* define threads_per_block */
  threads_per_block_on_create_seq = atoi(argv[1]);
  threads_per_block_on_rank = atoi(argv[2]);
  threads_per_block_on_full_verify = atoi(argv[3]);

  /* initialize the verification arrays for a valid class */
  for(i=0; i<TEST_ARRAY_SIZE; i++){
    switch(CLASS){
      case 'S':
        test_index_array[i] = S_test_index_array[i];
        test_rank_array[i]  = S_test_rank_array[i];
        break;
      case 'A':
        test_index_array[i] = A_test_index_array[i];
        test_rank_array[i]  = A_test_rank_array[i];
        break;
      case 'W':
        test_index_array[i] = W_test_index_array[i];
        test_rank_array[i]  = W_test_rank_array[i];
        break;
      case 'B':
        test_index_array[i] = B_test_index_array[i];
        test_rank_array[i]  = B_test_rank_array[i];
        break;
      case 'C':
        test_index_array[i] = C_test_index_array[i];
        test_rank_array[i]  = C_test_rank_array[i];
        break;
      case 'D':
        test_index_array[i] = D_test_index_array[i];
        test_rank_array[i]  = D_test_rank_array[i];
        break;
    };
  }

  threads_per_block_on_rank_1=1;
  threads_per_block_on_rank_2=threads_per_block_on_rank;
  threads_per_block_on_rank_3=threads_per_block_on_rank;
  threads_per_block_on_rank_4=threads_per_block_on_rank;
  threads_per_block_on_rank_5=threads_per_block_on_rank;
  threads_per_block_on_rank_6=threads_per_block_on_rank;
  threads_per_block_on_rank_7=1;
  threads_per_block_on_full_verify_1=threads_per_block_on_full_verify;
  threads_per_block_on_full_verify_2=threads_per_block_on_full_verify;
  threads_per_block_on_full_verify_3=threads_per_block_on_full_verify;

  amount_of_work_on_create_seq=threads_per_block_on_create_seq*threads_per_block_on_create_seq;
  amount_of_work_on_rank_1=1;
  amount_of_work_on_rank_2=MAX_KEY;
  amount_of_work_on_rank_3=NUM_KEYS;
  amount_of_work_on_rank_4=threads_per_block_on_rank_4*threads_per_block_on_rank_4;
  amount_of_work_on_rank_5=threads_per_block_on_rank_5;
  amount_of_work_on_rank_6=threads_per_block_on_rank_6*threads_per_block_on_rank_6;
  amount_of_work_on_rank_7=1;
  amount_of_work_on_full_verify_1=NUM_KEYS;
  amount_of_work_on_full_verify_2=NUM_KEYS;
  amount_of_work_on_full_verify_3=NUM_KEYS;

  blocks_per_grid_on_create_seq=(ceil((double)(amount_of_work_on_create_seq)/(double)(threads_per_block_on_create_seq)));

  blocks_per_grid_on_rank_1=1;

  blocks_per_grid_on_rank_2=(ceil((double)(amount_of_work_on_rank_2)/(double)(threads_per_block_on_rank_2)));

  blocks_per_grid_on_rank_3=(ceil((double)(amount_of_work_on_rank_3)/(double)(threads_per_block_on_rank_3)));

  if(amount_of_work_on_rank_4 > MAX_KEY){amount_of_work_on_rank_4=MAX_KEY;}
  blocks_per_grid_on_rank_4=(ceil((double)(amount_of_work_on_rank_4)/(double)(threads_per_block_on_rank_4)));

  blocks_per_grid_on_rank_5=1;

  if(amount_of_work_on_rank_6 > MAX_KEY){amount_of_work_on_rank_6=MAX_KEY;}
  blocks_per_grid_on_rank_6=(ceil((double)(amount_of_work_on_rank_6)/(double)(threads_per_block_on_rank_6)));

  blocks_per_grid_on_rank_7=1;

  blocks_per_grid_on_full_verify_1=(ceil((double)(amount_of_work_on_full_verify_1)/(double)(threads_per_block_on_full_verify_1)));
  blocks_per_grid_on_full_verify_2=(ceil((double)(amount_of_work_on_full_verify_2)/(double)(threads_per_block_on_full_verify_2)));
  blocks_per_grid_on_full_verify_3=(ceil((double)(amount_of_work_on_full_verify_3)/(double)(threads_per_block_on_full_verify_3)));

  size_shared_data_on_rank_4=2*threads_per_block_on_rank_4;
  size_shared_data_on_rank_5=2*threads_per_block_on_rank_5;
  size_shared_data_on_full_verify_3=threads_per_block_on_full_verify_3;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<int, 1> key_array_device (SIZE_OF_BUFFERS);
  buffer<int, 1> key_buff1_device (MAX_KEY);
  buffer<int, 1> key_buff2_device (SIZE_OF_BUFFERS);
  buffer<int, 1> index_array_device (test_index_array, TEST_ARRAY_SIZE);
  buffer<int, 1> rank_array_device (test_rank_array, TEST_ARRAY_SIZE);
  buffer<int, 1> partial_verify_vals_device (TEST_ARRAY_SIZE);
  buffer<int, 1> passed_verification_device (1);
  buffer<int, 1> sum_device (threads_per_block_on_rank);

  /* generate random number sequence and subsequent keys on all procs */

  range<1> lws_create_seq (threads_per_block_on_create_seq);
  range<1> gws_create_seq (threads_per_block_on_create_seq * blocks_per_grid_on_create_seq);
  q.submit([&] (handler &cgh) {
    auto key = key_array_device.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class create_seq>(nd_range<1>(gws_create_seq, lws_create_seq), [=] (nd_item<1> item) {
      create_seq_gpu_kernel (
        item,
        key.get_pointer(),
        314159265.00, /* random number gen seed */
        1220703125.00, /* random number gen mult */
        blocks_per_grid_on_create_seq,
        amount_of_work_on_create_seq);
    });
  });

  /* reset verification counter */
  passed_verification = 0;

  q.submit([&] (handler &cgh) {
    auto acc = passed_verification_device.get_access<sycl_write>(cgh);
    cgh.copy(&passed_verification, acc);
  });

  range<1> lws_rank_1 (threads_per_block_on_rank_1);
  range<1> gws_rank_1 (threads_per_block_on_rank_1 * blocks_per_grid_on_rank_1);
  range<1> lws_rank_2 (threads_per_block_on_rank_2);
  range<1> gws_rank_2 (threads_per_block_on_rank_2 * blocks_per_grid_on_rank_2);
  range<1> lws_rank_3 (threads_per_block_on_rank_3);
  range<1> gws_rank_3 (threads_per_block_on_rank_3 * blocks_per_grid_on_rank_3);
  range<1> lws_rank_4 (threads_per_block_on_rank_4);
  range<1> gws_rank_4 (threads_per_block_on_rank_4 * blocks_per_grid_on_rank_4);
  range<1> lws_rank_5 (threads_per_block_on_rank_5);
  range<1> gws_rank_5 (threads_per_block_on_rank_5 * blocks_per_grid_on_rank_5);
  range<1> lws_rank_6 (threads_per_block_on_rank_6);
  range<1> gws_rank_6 (threads_per_block_on_rank_6 * blocks_per_grid_on_rank_6);
  range<1> lws_rank_7 (threads_per_block_on_rank_7);
  range<1> gws_rank_7 (threads_per_block_on_rank_7 * blocks_per_grid_on_rank_7);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for(iteration=1; iteration<=MAX_ITERATIONS; iteration++) {
    q.submit([&] (handler &cgh) {
      auto key = key_array_device.get_access<sycl_write>(cgh);
      auto vals = partial_verify_vals_device.get_access<sycl_write>(cgh);
      auto index = index_array_device.get_access<sycl_read>(cgh);
      cgh.parallel_for<class rank1>(nd_range<1>(gws_rank_1, lws_rank_1), [=] (nd_item<1> item) {
        rank_gpu_kernel_1 (
          item,
          key.get_pointer(),
          vals.get_pointer(),
          index.get_pointer(),
          iteration,
          blocks_per_grid_on_rank_1,
          amount_of_work_on_rank_1);
      });
    });

    q.submit([&] (handler &cgh) {
      auto key = key_buff1_device.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class rank2>(nd_range<1>(gws_rank_2, lws_rank_2), [=] (nd_item<1> item) {
        rank_gpu_kernel_2 (
          item,
          key.get_pointer(),
          blocks_per_grid_on_rank_2,
          amount_of_work_on_rank_2);
      });
    });

    q.submit([&] (handler &cgh) {
      auto key_out = key_buff1_device.get_access<sycl_read_write>(cgh);
      auto key_in = key_array_device.get_access<sycl_read>(cgh);
      cgh.parallel_for<class rank3>(nd_range<1>(gws_rank_3, lws_rank_3), [=] (nd_item<1> item) {
        rank_gpu_kernel_3 (
          item,
          key_out.get_pointer(),
          key_in.get_pointer(),
          blocks_per_grid_on_rank_3,
          amount_of_work_on_rank_3);
      });
    });

    q.submit([&] (handler &cgh) {
      auto src = key_buff1_device.get_access<sycl_read>(cgh);
      auto dst = key_buff1_device.get_access<sycl_read_write>(cgh);
      auto sum = sum_device.get_access<sycl_discard_write>(cgh);
      accessor<int, 1, sycl_read_write, access::target::local> smem (size_shared_data_on_rank_4, cgh);
      cgh.parallel_for<class rank4>(nd_range<1>(gws_rank_4, lws_rank_4), [=] (nd_item<1> item) {
      rank_gpu_kernel_4(
          item,
          smem.get_pointer(),
          src.get_pointer(),
          dst.get_pointer(),
          sum.get_pointer(),
          blocks_per_grid_on_rank_4,
          amount_of_work_on_rank_4);
      });
    });

    q.submit([&] (handler &cgh) {
      auto sum = sum_device.get_access<sycl_read_write>(cgh);
      accessor<int, 1, sycl_read_write, access::target::local> smem (size_shared_data_on_rank_5, cgh);
      cgh.parallel_for<class rank5>(nd_range<1>(gws_rank_5, lws_rank_5), [=] (nd_item<1> item) {
        rank_gpu_kernel_5 (
          item,
          smem.get_pointer(),
          sum.get_pointer(),
          sum.get_pointer(),
          blocks_per_grid_on_rank_5,
          amount_of_work_on_rank_5);
      });
    });

    q.submit([&] (handler &cgh) {
      auto src = key_buff1_device.get_access<sycl_read>(cgh);
      auto dst = key_buff1_device.get_access<sycl_write>(cgh);
      auto sum = sum_device.get_access<sycl_read>(cgh);
      cgh.parallel_for<class rank6>(nd_range<1>(gws_rank_6, lws_rank_6), [=] (nd_item<1> item) {
        rank_gpu_kernel_6(
          item,
          src.get_pointer(),
          dst.get_pointer(),
          sum.get_pointer(),
          blocks_per_grid_on_rank_6,
          amount_of_work_on_rank_6);
      });
    });

    q.submit([&] (handler &cgh) {
      auto vals = partial_verify_vals_device.get_access<sycl_read>(cgh);
      auto key = key_buff1_device.get_access<sycl_read>(cgh);
      auto rank = rank_array_device.get_access<sycl_read>(cgh);
      auto pass = passed_verification_device.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class rank7>(nd_range<1>(gws_rank_7, lws_rank_7), [=] (nd_item<1> item) {
        rank_gpu_kernel_7(
          item,
          vals.get_pointer(),
          key.get_pointer(),
          rank.get_pointer(),
          pass.get_pointer(),
          iteration,
          blocks_per_grid_on_rank_7,
          amount_of_work_on_rank_7);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the rank kernels %f (s)\n",
         (time * 1e-9f) / MAX_ITERATIONS);

  q.submit([&] (handler &cgh) {
    auto acc = passed_verification_device.get_access<sycl_read>(cgh);
    cgh.copy(acc, &passed_verification);
  });

  /* 
   * this tests that keys are in sequence: sorting of last ranked key seq
   * occurs here, but is an untimed operation                             
   */
  int size_aux = amount_of_work_on_full_verify_3/threads_per_block_on_full_verify_3;
  int size_memory_aux=sizeof(int)*size_aux;

  buffer<int, 1> memory_aux_device (size_aux); 

  /* full_verify_gpu_kernel_1 */
  range<1> lws_verify_1 (threads_per_block_on_full_verify_1);
  range<1> gws_verify_1 (blocks_per_grid_on_full_verify_1 * threads_per_block_on_full_verify_1);
  
  q.submit([&] (handler &cgh) {
    auto key_in = key_array_device.get_access<sycl_read>(cgh);
    auto key_out = key_buff2_device.get_access<sycl_read>(cgh);
    cgh.parallel_for<class verify1>(nd_range<1>(gws_verify_1, lws_verify_1), [=] (nd_item<1> item) {
      full_verify_gpu_kernel_1(
        item,
        key_in.get_pointer(),
        key_out.get_pointer(),
        blocks_per_grid_on_full_verify_1,
        amount_of_work_on_full_verify_1);
    });
  });

  /* full_verify_gpu_kernel_2 */
  range<1> lws_verify_2 (threads_per_block_on_full_verify_2);
  range<1> gws_verify_2 (blocks_per_grid_on_full_verify_2 * threads_per_block_on_full_verify_2);
  
  q.submit([&] (handler &cgh) {
    auto key_in = key_buff2_device.get_access<sycl_read>(cgh);
    auto index = key_buff1_device.get_access<sycl_read_write>(cgh);
    auto key_out = key_array_device.get_access<sycl_write>(cgh);
    cgh.parallel_for<class verify2>(nd_range<1>(gws_verify_2, lws_verify_2), [=] (nd_item<1> item) {
      full_verify_gpu_kernel_2(
        item,
        key_in.get_pointer(),
        index.get_pointer(),
        key_out.get_pointer(),
        blocks_per_grid_on_full_verify_2,
        amount_of_work_on_full_verify_2);
    });
  });

  /* full_verify_gpu_kernel_3 */
  range<1> lws_verify_3 (threads_per_block_on_full_verify_3);
  range<1> gws_verify_3 (blocks_per_grid_on_full_verify_3 * threads_per_block_on_full_verify_3);
  
  q.submit([&] (handler &cgh) {
    auto key = key_array_device.get_access<sycl_read>(cgh);
    auto aux = memory_aux_device.get_access<sycl_discard_write>(cgh);
    accessor<int, 1, sycl_read_write, access::target::local> smem (size_shared_data_on_full_verify_3, cgh);
    cgh.parallel_for<class verify3>(nd_range<1>(gws_verify_3, lws_verify_3), [=] (nd_item<1> item) {
      full_verify_gpu_kernel_3(
        item,
        smem.get_pointer(),
        key.get_pointer(),
        aux.get_pointer(),
        blocks_per_grid_on_full_verify_3,
        amount_of_work_on_full_verify_3);
    });
  });

  /* reduce on cpu */
  int j = 0;
  int* memory_aux_host=(int*)malloc(size_memory_aux);
  q.submit([&] (handler &cgh) {
    auto aux = memory_aux_device.get_access<sycl_read>(cgh);
    cgh.copy(aux, memory_aux_host);
  }).wait();

  for(i=0; i<size_aux; i++){
    j += memory_aux_host[i];
  }  

  if(j!=0){
    printf( "Full_verify: number of keys out of sort: %ld\n", (long)j );
  }else{
    passed_verification++;
  }

  free(memory_aux_host);


  char gpu_config[256];
  char gpu_config_string[2048];
  sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
  strcpy(gpu_config_string, gpu_config);
  sprintf(gpu_config, "%29s\t%25d\n", " create", threads_per_block_on_create_seq);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, "%29s\t%25d\n", " rank", threads_per_block_on_rank);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, "%29s\t%25d\n", " verify", threads_per_block_on_full_verify);
  strcat(gpu_config_string, gpu_config);

  /* the final printout  */
  if(passed_verification != 5*MAX_ITERATIONS+1) {passed_verification = 0;}
  printf("%s\n", passed_verification ? "PASS" : "FAIL");

  return 0;  
}
