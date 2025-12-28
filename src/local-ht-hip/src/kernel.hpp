#include <stdio.h>
#define EMPTY 0xFFFFFFFF

//TODO: add this in a separate file for definitions
#ifdef DEBUG_GPU
#define DEBUG_PRINT_GPU 1
#endif

#ifdef DEBUG_CPU
#define DEBUG_PRINT_CPU 1
#endif

#define LASSM_MIN_QUAL 10
#define LASSM_MIN_HI_QUAL 20
#define LASSM_MIN_VIABLE_DEPTH 0.2
#define LASSM_MIN_EXPECTED_DEPTH 0.3
#define LASSM_RATING_THRES 0
//#define MAX_WALK_LEN 121+(2*251)
#define LASSM_MIN_KMER_LEN 21
#define LASSM_SHIFT_SIZE 8
#define LASSM_MAX_KMER_LEN 121
#define FULL_MASK 0xffffffff

struct cstr_type{
    char* start_ptr;
    int length;
    __device__ cstr_type(){}
    __device__ cstr_type(char* ptr, int len){
        start_ptr = ptr;
        length = len;
    }

    __device__ bool operator==(const cstr_type& in2){
        bool str_eq = true;
        if(length != EMPTY && in2.length != EMPTY)
            for(int i = 0; i < in2.length; i++){
                if(start_ptr[i] != in2.start_ptr[i]){
                    str_eq = false;
                    break;
                }
            }
        return (str_eq && (length == in2.length));
    }
};

__device__ void cstr_copy(cstr_type& str1, cstr_type& str2);

struct ExtCounts {
  uint32_t count_A;
  uint32_t count_C;
  uint32_t count_G;
  uint32_t count_T;

  __device__ void print(){
    printf("count_A:%d, count_C:%d, count_G:%d, count_T:%d\n", count_A, count_C, count_G, count_T);
  }

  __device__
  void inc(char ext, int count) {
    switch (ext) {
      case 'A':
        atomicAdd(&count_A,count);
        // count_A = (count_A > 65535) ? 65535 : count_A;
        break;
      case 'C':
        atomicAdd(&count_C,count);
        // count_C = (count_C > 65535) ? 65535 : count_C;
        break;
      case 'G':
        atomicAdd(&count_G,count);
        // count_G = (count_G > 65535) ? 65535 : count_G;
        break;
      case 'T':
        atomicAdd(&count_T,count);
        // count_T = (count_T > 65535) ? 65535 : count_T;
        break;
    }
  }
};


  struct MerBase {
    char base;
    uint32_t nvotes_hi_q, nvotes, rating;
    __device__ void print(){
      printf("base:%c, nvotes_hiq_q:%d, nvotes:%d, rating:%d\n", base, nvotes_hi_q, nvotes, rating);
    }

    __device__
    uint16_t get_base_rating(int depth) {
      double min_viable = max(LASSM_MIN_VIABLE_DEPTH * depth, 2.0);
      double min_expected_depth = max(LASSM_MIN_EXPECTED_DEPTH * depth, 2.0);
      if (nvotes == 0) return 0;
      if (nvotes == 1) return 1;
      if (nvotes < min_viable) return 2;
      if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q < min_viable) return 3;
      if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q >= min_viable) return 4;
      if (nvotes >= min_expected_depth && nvotes_hi_q < min_viable) return 5;
      if (nvotes >= min_expected_depth && min_viable < nvotes_hi_q && nvotes_hi_q < min_expected_depth) return 6;
      return 7;
    }
  };

struct MerFreqs {

  // how many times this kmer has occurred: don't need to count beyond 65536
  // count of high quality extensions and low quality extensions - structure comes from kmer_dht.hpp
  ExtCounts hi_q_exts, low_q_exts;
  // the final extensions chosen - A,C,G,T, or F,X
  char ext;
  // the count of the final extension
  int count;

    __device__
    bool comp_merbase(MerBase& elem1, MerBase& elem2){
        if(elem1.rating != elem2.rating)
            return elem1.rating > elem2.rating;
        if (elem1.nvotes_hi_q != elem2.nvotes_hi_q)
            return elem1.nvotes_hi_q > elem2.nvotes_hi_q;
        if (elem1.nvotes != elem2.nvotes)
            return elem1.nvotes > elem2.nvotes;

        return true;
    }

    __device__
    void sort_merbase(MerBase (&merbases)[4]){
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                if(comp_merbase(merbases[i], merbases[j])){
                    MerBase temp = merbases[i];
                    merbases[i] = merbases[j];
                    merbases[j] = temp;
                }
            }
        }
    }
  __device__
  void set_ext(int seq_depth) {
    // set extension similarly to how it is done with localassm in mhm
    MerBase mer_bases[4] = {{.base = 'A', .nvotes_hi_q = hi_q_exts.count_A, .nvotes = low_q_exts.count_A},
                            {.base = 'C', .nvotes_hi_q = hi_q_exts.count_C, .nvotes = low_q_exts.count_C},
                            {.base = 'G', .nvotes_hi_q = hi_q_exts.count_G, .nvotes = low_q_exts.count_G},
                            {.base = 'T', .nvotes_hi_q = hi_q_exts.count_T, .nvotes = low_q_exts.count_T}};
    for (int i = 0; i < 4; i++) {
      mer_bases[i].rating = mer_bases[i].get_base_rating(seq_depth);
    }

    // sort bases in descending order of quality
    sort_merbase(mer_bases);

    int top_rating = mer_bases[0].rating;
    int runner_up_rating = mer_bases[1].rating;
    //if (top_rating < runner_up_rating) 
    //DIE("top_rating ", top_rating, " < ", runner_up_rating, "\n");
    //the commented stuff above is handled by the assertion below on GPU
   // assert(top_rating >= runner_up_rating);// for now finding a way around for assertion
    if(top_rating < runner_up_rating)
      printf("******* ASSERTION FAILED IN sort_merbase************");
    int top_rated_base = mer_bases[0].base;
    ext = 'X';
    count = 0;
    // no extension (base = 0) if the runner up is close to the top rating
    // except, if rating is 7 (best quality), then all bases of rating 7 are forks
    if (top_rating > LASSM_RATING_THRES) {         // must have at least minViable bases
      if (top_rating <= 3) {    // must be uncontested
        if (runner_up_rating == 0) ext = top_rated_base;
      } else if (top_rating < 6) {
        if (runner_up_rating < 3) ext = top_rated_base;
      } else if (top_rating == 6) {  // viable and fair hiQ support
        if (runner_up_rating < 4) ext = top_rated_base;
      } else {                     // strongest rating trumps
        if (runner_up_rating < 7) {
          ext = top_rated_base;
        } else {
          if (mer_bases[2].rating == 7 || mer_bases[0].nvotes == mer_bases[1].nvotes) ext = 'F';
          else if (mer_bases[0].nvotes > mer_bases[1].nvotes) ext = mer_bases[0].base;
          else if (mer_bases[1].nvotes > mer_bases[0].nvotes) ext = mer_bases[1].base;
        }
      }
    }
    for (int i = 0; i < 4; i++) {
      if (mer_bases[i].base == ext) {
        count = mer_bases[i].nvotes;
        break;
      }
    }
  }
};


struct loc_ht{
    cstr_type key;
    MerFreqs val;
    __device__ loc_ht(cstr_type in_key, MerFreqs in_val){
      key = in_key;
      val = in_val;
    }
};

struct loc_ht_bool{
    cstr_type key;
    bool val;
    __device__ loc_ht_bool(cstr_type in_key, bool in_val){
      key = in_key;
      val = in_val;
    }
};

/* This function required some changes for compatibility
 * with the HIP library.  Synchronous shuffle functions do not
 * currently exist in HIP.  The functionality should be the same
 * as HIP wavefronts execute in lockstep, so as long as all
 * threads execute this function it should have the same
 * functionality.  That said, the functionality may not *strictly*
 * be identical.
 */
template <int WARP_SIZE>
__device__ int bcast_warp(int arg) {
  int laneId = threadIdx.x & WARP_SIZE;
  int value;
  if (laneId == 0)        // Note unused variable for
    value = arg;          // all threads except lane 0
  value = __shfl(value, 0);   // Get "value" from lane 0
  return value;
}

__device__ void print_mer(cstr_type& mer){
  for(int i = 0; i < mer.length; i++){
    printf("%c",mer.start_ptr[i]);
  }
  printf("\n");
}

__device__ void cstr_copy(cstr_type& str1, cstr_type& str2){

  for(int i = 0; i < str2.length; i++){
    str1.start_ptr[i] = str2.start_ptr[i];
  }
  str1.length = str2.length;
}

__device__ unsigned hash_func(cstr_type key, uint32_t max_size){
  unsigned hash, i;
  for(hash = i = 0; i < key.length; ++i)
  {
    hash += key.start_ptr[i];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash%max_size;//(hash & (HT_SIZE - 1));
}
#define MIX(h,k,m) { k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; }

__device__
uint32_t MurmurHashAligned2 (cstr_type key_in, uint32_t max_size)
{
  int len = key_in.length;
  char* key = key_in.start_ptr;
  const uint32_t m = 0x5bd1e995;
  const int r = 24;
  uint32_t seed = 0x3FB0BB5F;

  const unsigned char * data = (const unsigned char *)key;

  uint32_t h = seed ^ len;

  int align = (uint64_t)data & 3;

  if(align && (len >= 4))
  {
    /* Pre-load the temp registers  */

    uint32_t t = 0, d = 0;

    switch(align)
    {
      case 1: t |= data[2] << 16;
      case 2: t |= data[1] << 8;
      case 3: t |= data[0];
    }

    t <<= (8 * align);

    data += 4-align;
    len -= 4-align;

    int sl = 8 * (4-align);
    int sr = 8 * align;

    /* Mix */

    while(len >= 4)
    {
      d = *(uint32_t *)data;
      t = (t >> sr) | (d << sl);

      uint32_t k = t;

      MIX(h,k,m);

      t = d;

      data += 4;
      len -= 4;
    }

    /* Handle leftover data in temp registers  */

    d = 0;

    if(len >= align)
    {
      switch(align)
      {
        case 3: d |= data[2] << 16;
        case 2: d |= data[1] << 8;
        case 1: d |= data[0];
      }

      uint32_t k = (t >> sr) | (d << sl);
      MIX(h,k,m);

      data += align;
      len -= align;

      /* Handle tail bytes  */

      switch(len)
      {
        case 3: h ^= data[2] << 16;
        case 2: h ^= data[1] << 8;
        case 1: h ^= data[0];
          h *= m;
      };
    }
    else
    {
      switch(len)
      {
        case 3: d |= data[2] << 16;
        case 2: d |= data[1] << 8;
        case 1: d |= data[0];
        case 0: h ^= (t >> sr) | (d << sl);
          h *= m;
      }
    }

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h%max_size;
  }
  else
  {
    while(len >= 4)
    {
      uint32_t k = *(uint32_t *)data;

      MIX(h,k,m);

      data += 4;
      len -= 4;
    }

    /* Handle tail bytes  */

    switch(len)
    {
      case 3: h ^= data[2] << 16;
      case 2: h ^= data[1] << 8;
      case 1: h ^= data[0];
        h *= m;
    };

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h%max_size;
  }
}


__device__ void ht_insert(loc_ht* thread_ht, cstr_type kmer_key, MerFreqs mer_val, uint32_t max_size){
  unsigned hash_val = hash_func(kmer_key, max_size);
  //int count = 0; // for debugging
  while(true){
    int if_empty = thread_ht[hash_val].key.length; // length is set to some unimaginable number to indicate if its empty
    if(if_empty == EMPTY){ //the case where there is a key but no val, will not happen

      thread_ht[hash_val].key = kmer_key;
      thread_ht[hash_val].val = mer_val;
      return;
    }
    hash_val = (hash_val +1 ) % max_size;//(hash_val + 1) & (HT_SIZE-1);

  }
}
//overload for bool vals
__device__ void ht_insert(loc_ht_bool* thread_ht, cstr_type kmer_key, bool bool_val, uint32_t max_size){
  unsigned hash_val = hash_func(kmer_key, max_size);
  //int count = 0; // for debugging
  while(true){
    int if_empty = thread_ht[hash_val].key.length; // length is set to some unimaginable number to indicate if its empty
    if(if_empty == EMPTY){ //the case where there is a key but no val, will not happen
      thread_ht[hash_val].key = kmer_key;
      thread_ht[hash_val].val = bool_val;
      return;
    }
    hash_val = (hash_val +1 ) % max_size;//(hash_val + 1) & (HT_SIZE-1);
                 //count++; //for debugging

  }
}


__device__ 
loc_ht& ht_get(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size){
  unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
  unsigned orig_hash = hash_val;

  while(true){
    if(thread_ht[hash_val].key.length == EMPTY){
      return thread_ht[hash_val];
    }
    else if(thread_ht[hash_val].key == kmer_key){
      //printf("key found, returning\n");// keep this for debugging
      return thread_ht[hash_val];
    }
    hash_val = (hash_val +1 ) %max_size;//hash_val = (hash_val + 1) & (HT_SIZE -1);
    /*if(hash_val == orig_hash){ // loop till you reach the same starting positions and then return error
      printf("*****end reached, hashtable full*****\n"); // for debugging
      printf("*****end reached, hashtable full*****\n");
      printf("*****end reached, hashtable full*****\n");
    // return loc_ht(cstr_type(NULL,-1), MerFreqs());
    }*/
  }

}

//overload for bool vals
__device__ 
loc_ht_bool& ht_get(loc_ht_bool* thread_ht, cstr_type kmer_key, uint32_t max_size){
  unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
  unsigned orig_hash = hash_val;

  while(true){
    if(thread_ht[hash_val].key.length == EMPTY){
      return thread_ht[hash_val];
    }
    else if(thread_ht[hash_val].key == kmer_key){
      //printf("key found, returning\n");// keep this for debugging
      return thread_ht[hash_val];
    }
    hash_val = (hash_val +1 ) %max_size;//hash_val = (hash_val + 1) & (HT_SIZE -1);
    /*if(hash_val == orig_hash){ // loop till you reach the same starting positions and then return error
      printf("*****end reached, hashtable full*****\n"); // for debugging
      printf("*****end reached, hashtable full*****\n");
      printf("*****end reached, hashtable full*****\n");
    // return loc_ht(cstr_type(NULL,-1), MerFreqs());
    }*/
  }

}

/* This function required some significant changes due to lack
 * of support for certain functions in the HIP library.  The
 * __match_any_sync, __activemask, and __syncwarp functions do
 * not exist within HIP.  As such, the intended thread divergent
 * and selective synchronization behavior is not possible with
 * the current implementation.  Any access to this function as
 * it currently stands should be assumed to contain all of the
 * threads in the warp operating in lock step.
 */
__device__ 
loc_ht& ht_get_atomic(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size){
  unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
  unsigned orig_hash = hash_val;
  int done = 0;
  int prev;

  while(true){
    if(__all(done))
      return thread_ht[hash_val];

    if(!done) {
      prev = atomicCAS(&thread_ht[hash_val].key.length, EMPTY, kmer_key.length);

      // This function doesn't exist in HIP
      //int mask = __match_any_sync(__activemask(), (unsigned long long)&thread_ht[hash_val]); // all the threads in the warp which have same address

      if(prev == EMPTY){
        thread_ht[hash_val].key.start_ptr = kmer_key.start_ptr;
        thread_ht[hash_val].val = {.hi_q_exts = {0}, .low_q_exts = {0}, .ext = 0, .count = 0};
      }
    }

    // This function doesn't exist in HIP
    //__syncwarp(mask);

    if(!done) {
      if(prev != EMPTY && thread_ht[hash_val].key == kmer_key){
        //printf("key found, returning\n");// keep this for debugging
        done = 1;
        //return thread_ht[hash_val];
      }else if (prev == EMPTY){
        done = 1;
        //return thread_ht[hash_val];
      }
    }
    if(__all(done))
      return thread_ht[hash_val];
    if(!done) {
      hash_val = (hash_val + 1) % max_size;//hash_val = (hash_val + 1) & (HT_SIZE -1);
      if(hash_val == orig_hash){ // loop till you reach the same starting positions and then return error
        printf("*****end reached, hashtable full*****\n"); // for debugging
        printf("*****end reached, hashtable full*****\n");
        printf("*****end reached, hashtable full*****\n");
        done = 1; // We will return the current hash for now (though incorrect)
      }
    }
  }
}

__device__ char walk_mers(loc_ht* thrd_loc_ht, loc_ht_bool* thrd_ht_bool, uint32_t max_ht_size, int& mer_len, cstr_type& mer_walk_temp, cstr_type& longest_walk, cstr_type& walk, const int idx, int max_walk_len){
  char walk_result = 'X';
#ifdef DEBUG_PRINT_GPU
  int test = 1;
#endif
  for( int nsteps = 0; nsteps < max_walk_len; nsteps++){
    //check if there is a cycle in graph
    loc_ht_bool &temp_mer_loop = ht_get(thrd_ht_bool, mer_walk_temp, max_walk_len);
    if(temp_mer_loop.key.length == EMPTY){ // if the mer has not been visited, add it to the table and mark visited
      temp_mer_loop.key = mer_walk_temp;
      temp_mer_loop.val = true;
    }else{
      walk_result = 'R'; // if the table already contains this mer then a cycle exits, return the walk with repeat.
#ifdef DEBUG_PRINT_GPU
      if(idx == test)
        printf("breaking at cycle found, res: %c\n", walk_result);
#endif
      break;
    }

    loc_ht &temp_mer = ht_get(thrd_loc_ht, mer_walk_temp, max_ht_size);
    if(temp_mer.key.length == EMPTY){//if mer is not found then dead end reached, terminate the walk
      walk_result = 'X';
#ifdef DEBUG_PRINT_GPU
      if(idx == test)
        printf("breaking at mer not found,res: %c\n", walk_result);
#endif
      break;
    }
    char ext = temp_mer.val.ext;
    if(ext == 'F' || ext == 'X'){ // if the table points that ext is fork or dead end the terminate the walk
      walk_result = ext;
#ifdef DEBUG_PRINT_GPU
      if(idx == test){
        printf("breaking at dead end, res: %c\n", walk_result);
        printf("Mer Looked up:\n");
        print_mer(mer_walk_temp);
        printf("ext:%c\n",temp_mer.val.ext);
        printf("walk with mer_len:%d\n", mer_len);
        print_mer(walk);
      }
#endif
      break;
    }

#ifdef DEBUG_PRINT_GPU
    if(test == idx){
      printf("Mer Looked up:\n");
      print_mer(mer_walk_temp);
      printf("ext:%c\n",temp_mer.val.ext);
      printf("walk with mer_len:%d\n", mer_len);
      // print_mer(walk);
    }
#endif

    mer_walk_temp.start_ptr = mer_walk_temp.start_ptr + 1; // increment the mer pointer and append the ext
    mer_walk_temp.start_ptr[mer_walk_temp.length-1] = ext; // walk pointer points at the end of initial mer point.
    if (ext != 0) walk.length++;


  }

#ifdef DEBUG_PRINT_GPU
  if(idx == test)
    for (int k = 0; k < max_walk_len; k++) {
      if(thrd_ht_bool[k].key.length != EMPTY){
        printf("from bool ht:\n");
        print_mer(thrd_ht_bool[k].key);
        printf("Bool:%d\n",thrd_ht_bool[k].val);
      }
    }
#endif
  return walk_result;
}

template <int WARP_SIZE>
__device__ 
void count_mers(loc_ht* thrd_loc_ht, char* loc_r_reads, uint32_t max_ht_size, char* loc_r_quals, uint32_t* reads_r_offset, uint32_t& r_rds_cnt, 
    uint32_t* rds_count_r_sum, double& loc_ctg_depth, int& mer_len, uint32_t& qual_offset, int64_t& excess_reads, const long int idx){
  const int lane_id = threadIdx.x%WARP_SIZE;
  cstr_type read;
  cstr_type qual;
  uint32_t running_sum_len = 0;
  // #ifdef DEBUG_PRINT_GPU
  //int test = 1;

  //if(idx == 1 && threadIdx.x%WARP_SIZE == 0)
  //    to_print = true;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  // if(DEBUG_PRINT_GPU && idx == test)
  // #endif
  for(int i = 0; i < r_rds_cnt; i++){
    // #ifdef DEBUG_PRINT_GPU
    // if(DEBUG_PRINT_GPU && idx == test)
    //if(idx == test)
    // #endif
    read.start_ptr = loc_r_reads + running_sum_len;
    qual.start_ptr = loc_r_quals + running_sum_len;
    if(i == 0){
      if(idx == 0){
        read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];
        qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];
        // #ifdef DEBUG_PRINT_GPU
        // if(DEBUG_PRINT_GPU && idx == test)
        //if(idx == test)
        // #endif
      }
      else{  
        //printf("idx:%d, r_rdsx_cnt: %d, i: %d \n", idx, r_rds_cnt, i); 
        //printf("i:%d, rds_count:%d, idx:%d, thread:%d, blk:%d\n", i, r_rds_cnt, idx, tid, bid); 
        // #ifdef DEBUG_PRINT_GPU
        // if(DEBUG_PRINT_GPU && idx == test)
        //if(idx == test)
        // #endif
        if(rds_count_r_sum[idx - 1] == 0){
          read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];
          qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];                    
        }else{
          read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx - 1] -1)];
          qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx - 1] -1)];
        }
      }
    }
    else{
      read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + (i-1)];
      qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + (i-1)];
      // #ifdef DEBUG_PRINT_GPU
      // if(DEBUG_PRINT_GPU && idx == test)
      //if(idx == test)
      // #endif

    }
    // #ifdef DEBUG_PRINT_GPU
    // if(DEBUG_PRINT_GPU && idx == test){
    //if(idx == test){
    // #endif
    if (mer_len > read.length) // skip the read which is smaller than merlen
      continue;
    int num_mers = read.length - mer_len;

    for( int start = lane_id; start < num_mers; start+=WARP_SIZE){
      //TODO: on cpu side add a check that if a certain read contains 'N', 
      cstr_type mer(read.start_ptr + start, mer_len);

      loc_ht &temp_Mer = ht_get_atomic(thrd_loc_ht, mer, max_ht_size);

      int ext_pos = start + mer_len;
      //  assert(ext_pos < (int)read.length); // TODO: verify that assert works on gpu, for now commenting it out and replacing with printf
      if(ext_pos >= (int) read.length)
        printf("*********ASSERTION FAILURE IN COUNT_MERS****");
      char ext = read.start_ptr[ext_pos];
      if (ext == 'N') continue; // TODO: why the redundant check?
      int qual_diff = qual.start_ptr[ext_pos] - qual_offset;
      if (qual_diff >= LASSM_MIN_QUAL) temp_Mer.val.low_q_exts.inc(ext, 1);
      if (qual_diff >= LASSM_MIN_HI_QUAL) temp_Mer.val.hi_q_exts.inc(ext, 1);
    }
    // This function does not exist in HIP
    //__syncwarp();
    running_sum_len += read.length; // right before the for loop ends, update the prev_len to offset next read correctly
  }
  // This function does not exist in HIP
  //__syncwarp();
  for(auto k = lane_id; k < max_ht_size; k+=WARP_SIZE){
    if(thrd_loc_ht[k].key.length != EMPTY){
      thrd_loc_ht[k].val.set_ext(loc_ctg_depth);
    }
  }
  // This function does not exist in HIP
  //__syncwarp();
#ifdef DEBUG_PRINT_GPU
  int test = 1;
  if(idx == test){
    if(lane_id == 0)    
      for(int k = 0; k < max_ht_size; k++){
        //{
        if( thrd_loc_ht[k].key.length != EMPTY){
          printf("from ht:\n");
          print_mer(thrd_loc_ht[k].key);
          printf("MerFreq.ext:%c, MerFreq.count:%d\n",thrd_loc_ht[k].val.ext,thrd_loc_ht[k].val.count);
          thrd_loc_ht[k].val.hi_q_exts.print();
          thrd_loc_ht[k].val.low_q_exts.print();
        }
        //}
      }
  }
#endif
  // This function does not exist in HIP
  //__syncwarp();
  }

  //same kernel will be used for right and left walks
template <int WARP_SIZE>
__global__ void iterative_walks_kernel(
      uint32_t*  cid,
      uint32_t*  ctg_offsets,
      char*  contigs,
      char*  reads_r,
      char*  quals_r,
      uint32_t*  reads_r_offset,
      uint32_t*  rds_count_r_sum, 
      double*  ctg_depth,
      loc_ht*  global_ht,
      uint32_t*  prefix_ht,
      loc_ht_bool*  global_ht_bool,
      int kmer_len, uint32_t max_mer_len_off,
      uint32_t * term_counts,
      int64_t num_walks, int64_t max_walk_len, 
      int64_t sum_ext, int32_t max_read_size, int32_t max_read_count, uint32_t qual_offset,
      char*  longest_walks,
      char*  mer_walk_temp,
      uint32_t*  final_walk_lens,
      int tot_ctgs)
  {
    const long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const long int warp_id_glb = idx/WARP_SIZE;
    const long int lane_id = threadIdx.x%WARP_SIZE;
    if(warp_id_glb < tot_ctgs){ // so that no out of bound accesses 
      cstr_type loc_ctg;
      char *loc_r_reads, *loc_r_quals;
      uint32_t r_rds_cnt;
      loc_ht* loc_mer_map;
      uint32_t ht_loc_size;
      loc_ht_bool* loc_bool_map;
      double loc_ctg_depth;
      int64_t excess_reads;
      uint32_t max_ht_size = 0;
      char* longest_walk_loc;
      char* loc_mer_walk_temp;

      int min_mer_len = LASSM_MIN_KMER_LEN;
      int max_mer_len = LASSM_MAX_KMER_LEN;

      int active = 1;

      //the warp portion is for HT phase only so mapping only reads related data based on warp id
      if(warp_id_glb == 0){
        loc_ctg.start_ptr = contigs;
        loc_ctg.length = ctg_offsets[warp_id_glb];
        loc_bool_map = global_ht_bool + warp_id_glb * max_walk_len;
        longest_walk_loc = longest_walks + warp_id_glb * max_walk_len;
        loc_mer_walk_temp = mer_walk_temp + warp_id_glb * (max_walk_len + max_mer_len_off);
        r_rds_cnt = rds_count_r_sum[warp_id_glb];
        loc_r_reads = reads_r;
        loc_r_quals = quals_r;
        loc_mer_map = global_ht;
        ht_loc_size = prefix_ht[warp_id_glb];
        loc_ctg_depth = ctg_depth[warp_id_glb];
      }else{
        loc_ctg.start_ptr = contigs + ctg_offsets[warp_id_glb-1];
        loc_ctg.length = ctg_offsets[warp_id_glb] - ctg_offsets[warp_id_glb - 1];
        loc_bool_map = global_ht_bool + warp_id_glb * max_walk_len;
        longest_walk_loc = longest_walks + warp_id_glb * max_walk_len;
        loc_mer_walk_temp = mer_walk_temp + warp_id_glb * (max_walk_len + max_mer_len_off);
        loc_ctg_depth = ctg_depth[warp_id_glb];
        r_rds_cnt = rds_count_r_sum[warp_id_glb] - rds_count_r_sum[warp_id_glb - 1];
        if (rds_count_r_sum[warp_id_glb - 1] == 0)
          loc_r_reads = reads_r;
        else
          loc_r_reads = reads_r + reads_r_offset[rds_count_r_sum[warp_id_glb - 1] - 1]; // you want to start from where previous contigs, last read ends.

        if (rds_count_r_sum[warp_id_glb - 1] == 0)
          loc_r_quals = quals_r;
        else
          loc_r_quals = quals_r + reads_r_offset[rds_count_r_sum[warp_id_glb - 1] - 1]; // you want to start from where previous contigs, last read ends.

        loc_mer_map = global_ht + prefix_ht[warp_id_glb - 1];
        ht_loc_size = prefix_ht[warp_id_glb] - prefix_ht[warp_id_glb - 1];
      }

      max_ht_size = ht_loc_size;
      max_mer_len = min(max_mer_len, loc_ctg.length);

      cstr_type longest_walk_thread(longest_walk_loc,0);

      //main for loop
      int shift = 0;

      for(int mer_len = kmer_len; mer_len >= min_mer_len && mer_len <= max_mer_len; mer_len += shift){
        if(warp_id_glb < tot_ctgs){ // all warps within this range can go in execute count mers, for walk_mers only the lane 0 of each warp does the work
          for(uint32_t k = lane_id; k < max_ht_size; k+=WARP_SIZE){ // resetting hash table in parallel with warps
            loc_mer_map[k].key.length = EMPTY;
          }
          count_mers<WARP_SIZE>(loc_mer_map, loc_r_reads, max_ht_size, loc_r_quals, reads_r_offset, r_rds_cnt, rds_count_r_sum, loc_ctg_depth, mer_len, qual_offset, excess_reads, warp_id_glb);//passing warp_id instead of idx now
          for(uint32_t k = lane_id; k < max_walk_len; k+=WARP_SIZE){ // resetting bool map for next go
            loc_bool_map[k].key.length = EMPTY;
          }
          if(lane_id == 0){ // this phase is processed by single thread of a warp
            cstr_type ctg_mer(loc_ctg.start_ptr + (loc_ctg.length - mer_len), mer_len);
            cstr_type loc_mer_walk(loc_mer_walk_temp, 0);
            cstr_copy(loc_mer_walk, ctg_mer);
            cstr_type walk(loc_mer_walk.start_ptr + mer_len, 0);

            char walk_res = walk_mers(loc_mer_map, loc_bool_map, max_ht_size, mer_len, loc_mer_walk, longest_walk_thread, walk, warp_id_glb, max_walk_len);
            if (walk.length > longest_walk_thread.length){ // this walk is longer than longest then copy it to longest walk
              cstr_copy(longest_walk_thread, walk);
            }
            if (walk_res == 'X') {
              // walk reaches a dead-end, downshift, unless we were upshifting
              if (shift == LASSM_SHIFT_SIZE) {
                active = 0;
                goto end;
                //break;
              }
              shift = -LASSM_SHIFT_SIZE;
            }else {
              if (shift == -LASSM_SHIFT_SIZE){
                active = 0;
                goto end;
                //break;
              }
              if (mer_len > loc_ctg.length){
                active = 0;
                goto end;
                //break;
              }
              shift = LASSM_SHIFT_SIZE;
            }

          }// lane id cond ended
end:
          // This function does not exist in HIP
          //__syncwarp(FULL_MASK);
          /*
          // This function does not exist in HIP
          unsigned mask = __activemask();
          unsigned active = mask & 1; // zero if lane 0 has returned
           */
          active = bcast_warp<WARP_SIZE>(active);
          if(active == 0) break; // return if lane 0 has returned
          shift = bcast_warp<WARP_SIZE>(shift);
        }//warp id cond end

      }
      if(lane_id == 0){
        if(longest_walk_thread.length > 0){
          final_walk_lens[warp_id_glb] = longest_walk_thread.length;
        }else{
          final_walk_lens[warp_id_glb] = 0;
        }
      }
    }//end if to check if idx exceeds contigs
  }

