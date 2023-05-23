#define R23 sycl::pow(0.5, 23.0)
#define R46 (R23*R23)
#define T23 sycl::pow(2.0, 23.0)
#define T46 (T23*T23)

#define syncthreads() item.barrier(sycl::access::fence_space::local_space);

inline int atomicAdd(int &val, int operand) 
{
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  return atm.fetch_add(operand);
}

double randlc_device(double* X, const double* A)
{
  double T1, T2, T3, T4;
  double A1;
  double A2;
  double X1;
  double X2;
  double Z;
  int j;

  /*
   * --------------------------------------------------------------------
   * break A into two parts such that A = 2^23 * A1 + A2 and set X = N.
   * --------------------------------------------------------------------
   */
  T1 = R23 * *A;
  j  = T1;
  A1 = j;
  A2 = *A - T23 * A1;

  /*
   * --------------------------------------------------------------------
   * break X into two parts such that X = 2^23 * X1 + X2, compute
   * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
   * X = 2^23 * Z + A2 * X2  (mod 2^46). 
   * --------------------------------------------------------------------
   */
  T1 = R23 * *X;
  j  = T1;
  X1 = j;
  X2 = *X - T23 * X1;
  T1 = A1 * X2 + A2 * X1;

  j  = R23 * T1;
  T2 = j;
  Z = T1 - T23 * T2;
  T3 = T23 * Z + A2 * X2;
  j  = R46 * T3;
  T4 = j;
  *X = T3 - T46 * T4;

  return(R46 * *X);
}

double find_my_seed_device(
    int kn,
    int np,
    long nn,
    double s,
    double a)
{
  double t1,t2;
  long mq,nq,kk,ik;

  if(kn==0) return s;

  mq = (nn/4 + np - 1) / np;
  nq = mq * 4 * kn;

  t1 = s;
  t2 = a;
  kk = nq;
  while(kk > 1){
    ik = kk / 2;
    if(2*ik==kk){
      (void)randlc_device(&t2, &t2);
      kk = ik;
    }else{
      (void)randlc_device(&t1, &t2);
      kk = kk - 1;
    }
  }
  (void)randlc_device(&t1, &t2);

  return(t1);
}

void create_seq_gpu_kernel(
    sycl::nd_item<1> &item,
    int* key_array,
    double seed,
    double a,
    int number_of_blocks,
    int amount_of_work)
{
  double x, s;
  double an = a;
  int i, k;
  int k1, k2;
  int myid, num_procs;
  int mq;

  myid = item.get_global_id(0);
  num_procs = amount_of_work;

  mq = (NUM_KEYS + num_procs - 1) / num_procs;
  k1 = mq * myid;
  k2 = k1 + mq;
  if(k2 > NUM_KEYS) k2 = NUM_KEYS;

  s = find_my_seed_device(myid, num_procs, (long)4*NUM_KEYS, seed, an);

  k = MAX_KEY/4;

  for(i=k1; i<k2; i++){
    x = randlc_device(&s, &an);
    x += randlc_device(&s, &an);
    x += randlc_device(&s, &an);
    x += randlc_device(&s, &an);  
    key_array[i] = k*x;
  }
}

void full_verify_gpu_kernel_1(
    sycl::nd_item<1> &item,
    const int*__restrict key_array,
    int*__restrict key_buff2,
    int number_of_blocks,
    int amount_of_work)
{
  int i = item.get_global_id(0);
  key_buff2[i] = key_array[i];
}

void full_verify_gpu_kernel_2(
    sycl::nd_item<1> &item,
    const int* __restrict key_buff2,
    int*__restrict  key_buff_ptr_global,
    int*__restrict  key_array,
    int number_of_blocks,
    int amount_of_work)
{    
  int value = key_buff2[item.get_global_id(0)];
  int index = atomicAdd(key_buff_ptr_global[value], -1) - 1;
  key_array[index] = value;
}

void full_verify_gpu_kernel_3(
    sycl::nd_item<1> &item,
    int *__restrict shared_data,
    const int *__restrict key_array,
    int *__restrict global_aux,
    int number_of_blocks,
    int amount_of_work)
{

  int bid = item.get_group(0);
  int lid = item.get_local_id(0);
  int size = item.get_local_range(0);

  int i = (bid*size+lid) + 1;

  if(i < NUM_KEYS){
    if(key_array[i-1] > key_array[i]) 
      shared_data[lid]=1;
    else
      shared_data[lid]=0;
  }else
    shared_data[lid]=0;

  syncthreads();

  for(i=size/2; i>0; i>>=1) {
    if(lid<i)
      shared_data[lid] += shared_data[lid+i];
    syncthreads();
  }

  if(lid==0) global_aux[bid]=shared_data[0];
}


void rank_gpu_kernel_1(
    sycl::nd_item<1> &item,
    int*__restrict key_array,
    int*__restrict partial_verify_vals,
    const int*__restrict test_index_array,
    int iteration,
    int number_of_blocks,
    int amount_of_work)
{
  key_array[iteration] = iteration;
  key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;
  /*
   * --------------------------------------------------------------------
   * determine where the partial verify test keys are, 
   * --------------------------------------------------------------------
   * load into top of array bucket_size  
   * --------------------------------------------------------------------
   */
#pragma unroll
  for(int i=0; i<TEST_ARRAY_SIZE; i++){
    partial_verify_vals[i] = key_array[test_index_array[i]];
  }
}

void rank_gpu_kernel_2(
    sycl::nd_item<1> &item,
    int* key_buff1,
    int number_of_blocks,
    int amount_of_work)
{
  key_buff1[item.get_global_id(0)] = 0;
}

void rank_gpu_kernel_3(
    sycl::nd_item<1> &item,
    int*__restrict key_buff_ptr,
    const int*__restrict key_buff_ptr2,
    int number_of_blocks,
    int amount_of_work)
{
  /*
   * --------------------------------------------------------------------
   * in this section, the keys themselves are used as their 
   * own indexes to determine how many of each there are: their
   * individual population  
   * --------------------------------------------------------------------
   */
  
  atomicAdd(key_buff_ptr[key_buff_ptr2[item.get_global_id(0)]], 1);
}

void rank_gpu_kernel_4(
    sycl::nd_item<1> &item,
    int *__restrict shared_data,
    const int*__restrict source,
    int*__restrict destiny,
    int*__restrict sum,
    int number_of_blocks,
    int amount_of_work)
{

  int bid = item.get_group(0);
  int lid = item.get_local_id(0);
  int size = item.get_local_range(0);

  shared_data[lid] = 0;
  int position = size + lid;

  int factor = MAX_KEY / number_of_blocks;
  int start = factor * bid;
  int end = start + factor;

  for(int i=start; i<end; i+=size){
    shared_data[position] = source[i + lid];

    for(uint offset=1; offset<size; offset<<=1){
      syncthreads();
      int t = shared_data[position] + shared_data[position - offset];
      syncthreads();
      shared_data[position] = t;
    }

    int prv_val = (i == start) ? 0 : destiny[i - 1];
    destiny[i + lid] = shared_data[position] + prv_val;
  }

  syncthreads();
  if(lid==0) sum[bid] = destiny[end-1];
}

void rank_gpu_kernel_5(
    sycl::nd_item<1> &item,
    int *__restrict shared_data,
    const int*__restrict source,
    int*__restrict destiny,
    int number_of_blocks,
    int amount_of_work)
{
  int lid = item.get_local_id(0);
  int size = item.get_local_range(0);

  shared_data[lid] = 0;
  int position = size + lid;
  shared_data[position] = source[lid];

  for(uint offset=1; offset<size; offset<<=1) {
    syncthreads();
    int t = shared_data[position] + shared_data[position - offset];
    syncthreads();
    shared_data[position] = t;
  }

  syncthreads();

  destiny[lid] = shared_data[position - 1];
}

void rank_gpu_kernel_6(
    sycl::nd_item<1> &item,
    const int*__restrict source,
    int*__restrict destiny,
    const int*__restrict offset,
    int number_of_blocks,
    int amount_of_work)
{
  int bid = item.get_group(0);
  int lid = item.get_local_id(0);
  int size = item.get_local_range(0);

  int factor = MAX_KEY / number_of_blocks;
  int start = factor * bid;
  int end = start + factor;
  int sum = offset[bid];
  for(int i=start; i<end; i+=size)
    destiny[i + lid] = source[i + lid] + sum;
}

void rank_gpu_kernel_7(
    sycl::nd_item<1> &item,
    const int*__restrict partial_verify_vals,
    const int*__restrict key_buff_ptr,
    const int*__restrict test_rank_array,
    int*__restrict passed_verification_device,
    int iteration,
    int number_of_blocks,
    int amount_of_work)
{
  /*
   * --------------------------------------------------------------------
   * this is the partial verify test section 
   * observe that test_rank_array vals are
   * shifted differently for different cases
   * --------------------------------------------------------------------
   */
  int i, k;
  int passed_verification = 0;
  for(i=0; i<TEST_ARRAY_SIZE; i++){  
    /* test vals were put here on partial_verify_vals */                                           
    k = partial_verify_vals[i];          
    if(0<k && k<=NUM_KEYS-1){
      int key_rank = key_buff_ptr[k-1];
      int failed = 0;
      switch(CLASS){
        case 'S':
          if(i<=2){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'W':
          if(i<2){
            if(key_rank != test_rank_array[i]+(iteration-2))
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'A':
          if(i<=2){
            if(key_rank != test_rank_array[i]+(iteration-1))
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-(iteration-1))
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'B':
          if(i==1 || i==2 || i==4){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }
          else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'C':
          if(i<=2){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'D':
          if(i<2){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
      }
      if(failed==1){
        printf("Failed partial verification: iteration %d, test key %d\n", iteration, (int)i);
      }
    }
  }
  *passed_verification_device += passed_verification;
}
